"""Local stand-in for Lumen ``POST /Customer/v3/Ordering/orderRequest``.

Nothing in this module touches the network: the payload is validated against
a Pydantic mirror of the JSON schema in ``lumen_context.json`` and a
simulated 200 OK provisioning response is returned.
"""

import json
import logging
import secrets
import time
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
)
from pydantic.alias_generators import to_camel

import config

logger = logging.getLogger(__name__)

ENDPOINT = "POST /Customer/v3/Ordering/orderRequest"


class _ApiModel(BaseModel):
    """Base for API models: camelCase wire format, unknown keys rejected."""

    model_config = ConfigDict(
        alias_generator=to_camel, extra="forbid", frozen=True
    )


class Customer(_ApiModel):
    name: str = Field(min_length=1, max_length=120)


class ServiceLocation(_ApiModel):
    site_code: str = Field(pattern=r"^[A-Z]{3}[0-9]$")
    city: str = Field(min_length=1)
    country_code: str = Field(pattern=r"^[A-Z]{2}$")


class Port(_ApiModel):
    speed: Literal["1G", "10G"]
    quantity: StrictInt = Field(ge=1, le=16)
    bandwidth_mbps: StrictInt = Field(ge=1)


class CloudConnection(_ApiModel):
    provider: Literal["AWS", "GCP", "AZURE"]
    region: str = Field(min_length=1)
    product: str = Field(min_length=1)


class AddOns(_ApiModel):
    ddos_protection: StrictBool


class Billing(_ApiModel):
    pricing_model: Literal["HOURLY"]
    currency: Literal["USD"]
    hourly_rate_usd: float = Field(ge=0, allow_inf_nan=False)
    estimated_monthly_usd: float = Field(ge=0, allow_inf_nan=False)


class OrderRequest(_ApiModel):
    """Pydantic mirror of the ``orderRequest`` JSON schema."""

    quote_id: str = Field(pattern=r"^Q-[0-9A-F]{8}$")
    order_type: Literal["NEW_INSTALL"]
    product_family: Literal["NAAS_CLOUD_CONNECT"]
    customer: Customer
    service_location: ServiceLocation
    port: Port
    cloud_connection: CloudConnection
    add_ons: AddOns
    billing: Billing


def format_validation_errors(exc: ValidationError) -> str:
    """Render Pydantic errors as compact ``path: message`` lines."""
    return "\n".join(
        f"- {'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    )


def mock_lumen_provisioning_api(payload: dict[str, Any]) -> dict[str, Any]:
    """Simulate the Lumen ordering endpoint for a JSON order payload.

    Returns a 200-style SUCCESS body for valid payloads and a 400-style
    REJECTED body (instead of raising) for invalid ones, like a real API.
    """
    logger.info(
        "%s received payload:\n%s",
        ENDPOINT,
        json.dumps(payload, indent=2, default=str),
    )
    time.sleep(config.MOCK_API_LATENCY_SECONDS)  # simulated network latency

    try:
        order = OrderRequest.model_validate(payload)
    except ValidationError as exc:
        errors = format_validation_errors(exc)
        logger.warning("%s rejected payload:\n%s", ENDPOINT, errors)
        return {
            "status": "REJECTED",
            "http_status": 400,
            "message": "Payload failed orderRequest schema validation",
            "errors": errors,
        }

    response: dict[str, Any] = {
        "status": "SUCCESS",
        "http_status": 200,
        "transaction_id": f"LUMEN-NAAS-{secrets.randbelow(90_000) + 10_000}",
        "message": "Network provisioned",
        "quote_id": order.quote_id,
        "provisioned_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    logger.info("%s responded: %s", ENDPOINT, response)
    return response
