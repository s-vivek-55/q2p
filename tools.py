"""Deterministic context, inventory and pricing tools (no LLM, $0 cost).

All arithmetic lives here so prices are never hallucinated by a model.
"""

import math
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import config

PortSpeed = Literal["1G", "10G"]
CloudProvider = Literal["AWS", "GCP", "AZURE"]
Geo = Literal["NA", "EMEA", "APAC"]


class ContextError(RuntimeError):
    """The local Lumen context file is missing or malformed."""


class QuoteError(ValueError):
    """A quote cannot be produced for the requested combination."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Context models (mirror lumen_context.json) ---------------------------
class Location(_Frozen):
    city: str
    country_code: str
    site_code: str
    pop_name: str
    geo: Geo
    aliases: tuple[str, ...] = ()
    available_ports: dict[PortSpeed, int]
    cloud_regions: dict[CloudProvider, str]


class CloudPricing(_Frozen):
    product: str
    onramp_hourly_usd: dict[PortSpeed, float]
    egress_per_gb_usd: dict[Geo, float]


class OrderRequestApi(_Frozen):
    method: Literal["POST"]
    path: str
    request_schema: dict[str, Any]


class LumenContext(_Frozen):
    currency: Literal["USD"]
    hours_per_month: int = Field(gt=0)
    default_monthly_egress_gb: int = Field(ge=0)
    port_capacity_mbps: dict[PortSpeed, int]
    port_pricing_hourly_usd: dict[PortSpeed, float]
    ddos_addon_hourly_usd: dict[PortSpeed, float]
    cloud_providers: dict[CloudProvider, CloudPricing]
    locations: tuple[Location, ...]
    order_request_api: OrderRequestApi


# --- Quote models ---------------------------------------------------------
class LineItem(_Frozen):
    sku: str
    description: str
    quantity: int
    unit_hourly_usd: float
    hourly_usd: float


class CloudOption(_Frozen):
    provider: CloudProvider
    product: str
    region: str
    onramp_hourly_usd: float
    egress_hourly_usd: float
    effective_hourly_usd: float


class PortAvailability(_Frozen):
    available: bool
    in_stock: int
    requested: int
    capex_usd: float
    message: str


class Quote(_Frozen):
    quote_id: str
    customer_name: str
    site_code: str
    pop_name: str
    city: str
    country_code: str
    bandwidth_mbps: int
    port_speed: PortSpeed
    port_quantity: int
    cloud_provider: CloudProvider
    cloud_product: str
    cloud_region: str
    cloud_recommended: bool  # True when the optimizer picked the cloud
    cloud_options: tuple[CloudOption, ...]  # cheapest first
    ddos_protection: bool
    monthly_egress_gb: int
    line_items: tuple[LineItem, ...]
    total_hourly_usd: float
    estimated_monthly_usd: float


# --- Context access -------------------------------------------------------
@lru_cache(maxsize=4)
def load_context(path: Path = config.CONTEXT_PATH) -> LumenContext:
    """Load and validate the mock Lumen dataset (cached per path)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContextError(f"Cannot read context file '{path}': {exc}") from exc
    try:
        return LumenContext.model_validate_json(raw)
    except ValidationError as exc:
        raise ContextError(f"Context file '{path}' is invalid: {exc}") from exc


def find_location(ctx: LumenContext, query: str | None) -> Location | None:
    """Case-insensitive match on city, site code or alias."""
    if not query or not query.strip():
        return None
    needle = query.split(",")[0].strip().casefold()
    return next(
        (
            loc
            for loc in ctx.locations
            if needle
            in {
                loc.city.casefold(),
                loc.site_code.casefold(),
                *(alias.casefold() for alias in loc.aliases),
            }
        ),
        None,
    )


def select_port(ctx: LumenContext, bandwidth_mbps: int) -> tuple[PortSpeed, int]:
    """Smallest single port that fits, else N x the largest port speed."""
    if bandwidth_mbps <= 0:
        raise QuoteError("Bandwidth must be a positive number of Mbps.")
    by_capacity = sorted(ctx.port_capacity_mbps.items(), key=lambda kv: kv[1])
    for speed, capacity in by_capacity:
        if bandwidth_mbps <= capacity:
            return speed, 1
    speed, capacity = by_capacity[-1]
    return speed, math.ceil(bandwidth_mbps / capacity)


def check_port_availability(
    location: Location, speed: PortSpeed, quantity: int
) -> PortAvailability:
    """Simulated $0 CapEx check: idle on-net ports need no construction."""
    in_stock = location.available_ports.get(speed, 0)
    available = in_stock >= quantity
    message = (
        f"{in_stock} idle {speed} port(s) on-net at {location.site_code} "
        "— $0 CapEx, no construction required."
        if available
        else f"Only {in_stock} idle {speed} port(s) at {location.site_code} "
        f"({quantity} requested) — an off-net build would be required."
    )
    return PortAvailability(
        available=available,
        in_stock=in_stock,
        requested=quantity,
        capex_usd=0.0,
        message=message,
    )


def find_alternative_sites(
    ctx: LumenContext, location: Location, speed: PortSpeed, quantity: int
) -> list[Location]:
    """On-net sites in the same geography with enough idle capacity."""
    return [
        loc
        for loc in ctx.locations
        if loc.site_code != location.site_code
        and loc.geo == location.geo
        and loc.available_ports.get(speed, 0) >= quantity
    ]


def rank_cloud_options(
    ctx: LumenContext,
    location: Location,
    speed: PortSpeed,
    port_quantity: int,
    monthly_egress_gb: int,
) -> list[CloudOption]:
    """Effective hourly cost (on-ramp + amortised egress) per cloud, cheapest first."""

    def option(provider: CloudProvider) -> CloudOption:
        pricing = ctx.cloud_providers[provider]
        onramp = pricing.onramp_hourly_usd[speed] * port_quantity
        egress = (
            pricing.egress_per_gb_usd[location.geo]
            * monthly_egress_gb
            / ctx.hours_per_month
        )
        return CloudOption(
            provider=provider,
            product=pricing.product,
            region=location.cloud_regions[provider],
            onramp_hourly_usd=round(onramp, 4),
            egress_hourly_usd=round(egress, 4),
            effective_hourly_usd=round(onramp + egress, 4),
        )

    return sorted(
        (option(p) for p in ctx.cloud_providers if p in location.cloud_regions),
        key=lambda opt: opt.effective_hourly_usd,
    )


def _line(sku: str, description: str, quantity: int, unit: float) -> LineItem:
    return LineItem(
        sku=sku,
        description=description,
        quantity=quantity,
        unit_hourly_usd=round(unit, 4),
        hourly_usd=round(unit * quantity, 4),
    )


def calculate_quote(
    ctx: LumenContext,
    *,
    customer_name: str,
    location: Location,
    bandwidth_mbps: int,
    speed: PortSpeed,
    port_quantity: int,
    cloud_provider: CloudProvider | None,
    ddos_protection: bool,
    monthly_egress_gb: int,
) -> Quote:
    """Build the hourly Bill of Materials; auto-selects the cheapest cloud."""
    options = rank_cloud_options(
        ctx, location, speed, port_quantity, monthly_egress_gb
    )
    chosen = next(
        (o for o in options if cloud_provider in (None, o.provider)), None
    )
    if chosen is None:
        raise QuoteError(
            f"{cloud_provider} on-ramp is not available at {location.city}."
        )

    items = (
        _line(
            f"NAAS-PORT-{speed}",
            f"Lumen NaaS Ethernet port {speed} @ {location.site_code}",
            port_quantity,
            ctx.port_pricing_hourly_usd[speed],
        ),
        _line(
            f"ONRAMP-{chosen.provider}-{speed}",
            f"{chosen.product} on-ramp ({chosen.region})",
            port_quantity,
            ctx.cloud_providers[chosen.provider].onramp_hourly_usd[speed],
        ),
        _line(
            f"EGRESS-{chosen.provider}",
            f"Est. {chosen.provider} egress ({monthly_egress_gb:,} GB/mo)",
            1,
            chosen.egress_hourly_usd,
        ),
        *(
            [
                _line(
                    f"DDOS-{speed}",
                    "Lumen DDoS Mitigation add-on",
                    port_quantity,
                    ctx.ddos_addon_hourly_usd[speed],
                )
            ]
            if ddos_protection
            else []
        ),
        _line("ONNET-CAPEX", "On-net capacity (no construction)", 1, 0.0),
    )
    total = round(sum(item.hourly_usd for item in items), 4)

    return Quote(
        quote_id=f"Q-{uuid.uuid4().hex[:8].upper()}",
        customer_name=customer_name,
        site_code=location.site_code,
        pop_name=location.pop_name,
        city=location.city,
        country_code=location.country_code,
        bandwidth_mbps=bandwidth_mbps,
        port_speed=speed,
        port_quantity=port_quantity,
        cloud_provider=chosen.provider,
        cloud_product=chosen.product,
        cloud_region=chosen.region,
        cloud_recommended=cloud_provider is None,
        cloud_options=tuple(options),
        ddos_protection=ddos_protection,
        monthly_egress_gb=monthly_egress_gb,
        line_items=items,
        total_hourly_usd=total,
        estimated_monthly_usd=round(total * ctx.hours_per_month, 2),
    )
