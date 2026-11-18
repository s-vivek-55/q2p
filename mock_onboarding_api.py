"""Local stand-ins for the Lumen new-customer APIs (no network calls).

* ``GET  /Customer/v3/Serviceability?siteCode=`` - services sold at a site.
* ``POST /Customer/v3/Accounts``                - create a customer account.

Like ``mock_api.py`` they log, simulate latency, validate and respond.
"""

import logging
import secrets
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

import config
from mock_api import format_validation_errors
from tools import ContextError, Geo, load_context

logger = logging.getLogger(__name__)

SERVICES_PATH: Final[Path] = config.BASE_DIR / "lumen_services.json"
SERVICEABILITY_ENDPOINT: Final[str] = "GET /Customer/v3/Serviceability"
ACCOUNTS_ENDPOINT: Final[str] = "POST /Customer/v3/Accounts"
EMAIL_PATTERN: Final[str] = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
_LATENCY_SECONDS: Final[float] = 1.0


# --- Service catalogue (lumen_services.json) -------------------------------
class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ServiceOffer(_Frozen):
    code: str
    name: str
    description: str
    geos: tuple[Geo, ...]
    from_hourly_usd: float = Field(ge=0)
    sites: tuple[str, ...] = ()  # empty = every on-net site in ``geos``


class SizingRules(_Frozen):
    """Lumen site-sizing standards used to recommend bandwidth."""

    sqft_per_user: int = Field(gt=0)
    users_per_floor: int = Field(gt=0)
    mbps_per_user: float = Field(gt=0)
    peak_concurrency: float = Field(gt=0, le=1)
    headroom: float = Field(ge=1)
    switch_ports: int = Field(gt=0)
    redundancy_min_floors: int = Field(gt=0)


class ServiceCatalog(_Frozen):
    sizing: SizingRules
    services: tuple[ServiceOffer, ...]


@lru_cache(maxsize=1)
def load_service_catalog(path: Path = SERVICES_PATH) -> ServiceCatalog:
    try:
        return ServiceCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ContextError(f"Service catalogue '{path}' is unusable: {exc}") from exc


def mock_lumen_serviceability_api(site_code: str) -> dict[str, Any]:
    """Services, idle ports and cloud on-ramps available at an on-net site."""
    logger.info("%s?siteCode=%s", SERVICEABILITY_ENDPOINT, site_code)
    time.sleep(_LATENCY_SECONDS)
    location = next(
        (loc for loc in load_context().locations if loc.site_code == site_code),
        None,
    )
    if location is None:
        return {
            "status": "NOT_FOUND",
            "http_status": 404,
            "message": f"{site_code} is not a Lumen on-net site",
        }
    services = [
        {
            "code": offer.code,
            "name": offer.name,
            "description": offer.description,
            "fromHourlyUsd": offer.from_hourly_usd,
        }
        for offer in load_service_catalog().services
        if location.geo in offer.geos
        and (not offer.sites or site_code in offer.sites)
    ]
    logger.info("%s responded: %d services at %s",
                SERVICEABILITY_ENDPOINT, len(services), site_code)
    return {
        "status": "SUCCESS",
        "http_status": 200,
        "siteCode": site_code,
        "onNet": True,
        "availablePorts": dict(location.available_ports),
        "cloudOnRamps": dict(location.cloud_regions),
        "services": services,
    }


# --- Account creation -------------------------------------------------------
class _ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, extra="forbid", frozen=True
    )


class Contact(_ApiModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(pattern=EMAIL_PATTERN, max_length=254)


class ServiceAddress(_ApiModel):
    site_code: str = Field(pattern=r"^[A-Z]{3}[0-9]$")
    city: str = Field(min_length=1)
    country_code: str = Field(pattern=r"^[A-Z]{2}$")


class AccountRequest(_ApiModel):
    account_type: Literal["NEW_ENTERPRISE"]
    company_name: str = Field(min_length=1, max_length=120)
    contact: Contact
    service_address: ServiceAddress


def mock_lumen_create_account_api(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a new enterprise account; 201 on success, 400 on bad input."""
    time.sleep(_LATENCY_SECONDS)
    try:
        request = AccountRequest.model_validate(payload)
    except ValidationError as exc:
        errors = format_validation_errors(exc)
        logger.warning("%s rejected payload:\n%s", ACCOUNTS_ENDPOINT, errors)
        return {
            "status": "REJECTED",
            "http_status": 400,
            "message": "Account request failed validation",
            "errors": errors,
        }
    account_id = f"LUMEN-ACCT-{secrets.randbelow(900_000) + 100_000}"
    # Contact details are PII: log only company and site.
    logger.info("%s created %s for %s at %s", ACCOUNTS_ENDPOINT, account_id,
                request.company_name, request.service_address.site_code)
    return {
        "status": "SUCCESS",
        "http_status": 201,
        "account_id": account_id,
        "message": "Customer account created",
        "company_name": request.company_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
