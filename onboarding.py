"""New-customer onboarding: requirement intake, site sizing and plan finder.

Flow: free text -> ``extract_requirement`` (NLU model, optional form
prefill) -> validated ``NewCustomerRequirement`` -> ``build_proposal``
(serviceability API, site sizing, AWS/GCP/Azure plan comparison) ->
``create_account`` (accounts API) -> the existing
``agent.generate_provisioning_payload`` for the chosen plan.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent import (  # reuse the existing NLU plumbing (logging, cost, errors)
    AgentError,
    ChatMessage,
    NetworkIntent,
    Usage,
    _call_nlu,
    _extract_json_object,
)
from mock_onboarding_api import (
    EMAIL_PATTERN,
    SizingRules,
    load_service_catalog,
    mock_lumen_create_account_api,
    mock_lumen_serviceability_api,
)
from tools import (
    CloudProvider,
    Location,
    LumenContext,
    Quote,
    QuoteError,
    calculate_quote,
    check_port_availability,
    find_alternative_sites,
    find_location,
    select_port,
)

logger = logging.getLogger(__name__)


# --- Requirement models ------------------------------------------------------
class NewCustomerIntent(NetworkIntent):
    """LLM-extracted requirement (lenient: bad or missing values -> None)."""

    floors: int | None = None
    area_sqft: int | None = None
    contact_name: str | None = None
    contact_email: str | None = None

    @field_validator("floors", "area_sqft", mode="before")
    @classmethod
    def _positive(cls, value: Any) -> Any:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return round(value) if value > 0 else None
        return value


class NewCustomerRequirement(BaseModel):
    """Validated requirement captured from the intake form."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    company_name: str = Field(min_length=1, max_length=120)
    contact_name: str = Field(min_length=1, max_length=120)
    contact_email: str = Field(pattern=EMAIL_PATTERN, max_length=254)
    city: str = Field(min_length=1)
    floors: int = Field(ge=1, le=200)
    area_sqft: int = Field(ge=0, le=10_000_000)  # 0 = unknown, size by floors
    bandwidth_mbps: int = Field(ge=0, le=1_000_000)  # 0 = size it for me
    ddos_protection: bool
    cloud_provider: CloudProvider | None = None  # preference; all are compared
    monthly_egress_gb: int = Field(ge=0)


@dataclass(frozen=True)
class SiteSizing:
    estimated_users: int
    demand_mbps: int
    recommended_mbps: int
    floor_switches: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PlanOption:
    name: str  # "Recommended" or "Right-sized"
    quote: Quote


@dataclass(frozen=True)
class Proposal:
    requirement: NewCustomerRequirement
    location: Location
    sizing: SiteSizing
    services: tuple[dict[str, Any], ...]
    options: tuple[PlanOption, ...]  # cheapest first
    best: PlanOption | None
    warnings: tuple[str, ...]


# --- NLU prefill -------------------------------------------------------------
def _intake_prompt(ctx: LumenContext) -> str:
    cities = ", ".join(loc.city for loc in ctx.locations)
    return (
        "You are the new-customer intake engine of Lumen's NaaS Broker. "
        "Extract the prospect's office connectivity requirement.\n"
        f"Lumen on-net cities: {cities}. Clouds: AWS, GCP, AZURE.\n"
        "Respond with ONLY a JSON object with exactly these keys, using null "
        "when a value is not stated:\n"
        '"customer_name" (company), "contact_name", "contact_email", '
        '"source_city" (canonical on-net city when it matches), '
        '"floors" (integer), "area_sqft" (integer square feet; m2 x 10.764), '
        '"bandwidth_mbps" (integer Mbps, 1 Gbps = 1000; read "GHz" or '
        '"gigahertz" as Gbps; if several speeds are given use the largest), '
        '"cloud_provider", "ddos_protection" (true/false), '
        '"monthly_egress_gb", "reply" (one short sentence, no prices).'
    )


def extract_requirement(
    text: str, ctx: LumenContext
) -> tuple[NewCustomerIntent, Usage]:
    """Prefill the intake form from free text with the cheap NLU model."""
    raw, usage = _call_nlu(
        _intake_prompt(ctx), [ChatMessage(role="user", content=text)]
    )
    try:
        intent = NewCustomerIntent.model_validate(_extract_json_object(raw))
    except ValueError as exc:
        raise AgentError(
            "I couldn't interpret that description.", usage
        ) from exc
    logger.info("New-customer intent: %s", intent.model_dump(
        exclude={"contact_name", "contact_email"}))
    return intent, usage


# --- Sizing & plans (deterministic, $0) --------------------------------------
def size_site(req: NewCustomerRequirement, rules: SizingRules) -> SiteSizing:
    """Estimate users and peak demand from area (or floors) per Lumen rules."""
    by_area = req.area_sqft > 0
    users = (
        math.ceil(req.area_sqft / rules.sqft_per_user)
        if by_area
        else req.floors * rules.users_per_floor
    )
    demand = math.ceil(
        users * rules.mbps_per_user * rules.peak_concurrency * rules.headroom
    )
    per_floor = math.ceil(users / req.floors)
    switches = req.floors * math.ceil(per_floor / rules.switch_ports)
    notes = [
        f"{users:,} estimated users "
        + (f"({req.area_sqft:,} sq ft ÷ {rules.sqft_per_user} sq ft/user)"
           if by_area
           else f"({req.floors} floors × {rules.users_per_floor} users/floor)"),
        f"Peak demand ≈ {demand:,} Mbps ({rules.mbps_per_user:g} Mbps/user × "
        f"{rules.peak_concurrency:.0%} concurrency × "
        f"{rules.headroom:.0%} growth headroom)",
        f"In-building: {switches} × {rules.switch_ports}-port access "
        f"switch(es) across {req.floors} floor(s)",
    ]
    if req.floors >= rules.redundancy_min_floors:
        notes.append(
            "A diverse (dual-entry) building connection is recommended for "
            f"{req.floors} floors"
        )
    return SiteSizing(users, demand, max(req.bandwidth_mbps, demand),
                      switches, tuple(notes))


def build_proposal(ctx: LumenContext, req: NewCustomerRequirement) -> Proposal:
    """Check serviceability, size the site and price every cloud option."""
    location = find_location(ctx, req.city)
    if location is None:
        raise QuoteError(
            f"{req.city} is not a Lumen on-net location yet. Available: "
            + ", ".join(loc.city for loc in ctx.locations)
        )
    sizing = size_site(req, load_service_catalog().sizing)
    services = tuple(
        mock_lumen_serviceability_api(location.site_code).get("services", [])
    )

    warnings: list[str] = []
    if 0 < req.bandwidth_mbps < sizing.demand_mbps:
        warnings.append(
            f"Requested {req.bandwidth_mbps:,} Mbps is below the estimated "
            f"peak demand of {sizing.demand_mbps:,} Mbps, so plans are sized "
            "to the demand."
        )
    tiers = {"Recommended": sizing.recommended_mbps,
             "Right-sized": sizing.demand_mbps}

    options: list[PlanOption] = []
    seen_ports: set[tuple[str, int]] = set()
    for tier, mbps in tiers.items():
        speed, quantity = select_port(ctx, mbps)
        if (speed, quantity) in seen_ports:
            continue  # same port as a previous tier means the same price
        seen_ports.add((speed, quantity))
        availability = check_port_availability(location, speed, quantity)
        if not availability.available:
            alternatives = find_alternative_sites(ctx, location, speed, quantity)
            warnings.append(
                f"{tier} ({mbps:,} Mbps): {availability.message}"
                + (" Alternatives: " + ", ".join(a.city for a in alternatives)
                   if alternatives else "")
            )
            continue
        options += [
            PlanOption(tier, calculate_quote(
                ctx,
                customer_name=req.company_name,
                location=location,
                bandwidth_mbps=mbps,
                speed=speed,
                port_quantity=quantity,
                cloud_provider=provider,
                ddos_protection=req.ddos_protection,
                monthly_egress_gb=req.monthly_egress_gb,
            ))
            for provider in location.cloud_regions
        ]

    options.sort(key=lambda o: o.quote.estimated_monthly_usd)
    recommended = [o for o in options if o.name == "Recommended"]
    best = next(iter(recommended or options), None)
    logger.info("Proposal for %s @ %s: %d options, best=%s", req.company_name,
                location.site_code, len(options),
                best and (best.name, best.quote.cloud_provider))
    return Proposal(req, location, sizing, services, tuple(options), best,
                    tuple(warnings))


def create_account(
    req: NewCustomerRequirement, location: Location
) -> dict[str, Any]:
    """Register the prospect as a new Lumen customer (mock Accounts API)."""
    return mock_lumen_create_account_api({
        "accountType": "NEW_ENTERPRISE",
        "companyName": req.company_name,
        "contact": {"name": req.contact_name, "email": req.contact_email},
        "serviceAddress": {
            "siteCode": location.site_code,
            "city": location.city,
            "countryCode": location.country_code,
        },
    })
