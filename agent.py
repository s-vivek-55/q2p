"""LLM orchestration for the Zero-Touch NaaS Broker.

* ``config.NLU_MODEL`` (Chat Completions) extracts intent from the chat.
* Deterministic tools price the Bill of Materials and draw the topology.
* ``config.CODEX_MODEL`` (Responses API) compiles the approved quote into the
  orderRequest payload, which is self-healed against Pydantic validation and
  executed against the local mock API.

Models come from ``config.MODEL_PROFILE``: "dev" uses gpt-5.6-luna for both
roles (cheapest measured on the gateway); "prod" uses gpt-5-mini (NLU) and
gpt-5.3-codex (payload).
"""

import json
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, TypedDict

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

import config
from logging_setup import describe_api_key, key_shape, redact
from mock_api import (
    OrderRequest,
    format_validation_errors,
    mock_lumen_provisioning_api,
)
from tools import (
    CloudProvider,
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


# --- Public types ---------------------------------------------------------
class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    """Token usage and estimated spend of one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )

    @classmethod
    def for_model(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
        gateway_cost_usd: float | None = None,
    ) -> "Usage":
        """Prefer the cost the gateway reports; else estimate from the table."""
        if gateway_cost_usd is not None:
            return cls(input_tokens, output_tokens, gateway_cost_usd)
        # Unknown models are billed at the most expensive known rate.
        in_rate, out_rate = config.MODEL_PRICING_PER_MTOK.get(
            model, max(config.MODEL_PRICING_PER_MTOK.values())
        )
        cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
        return cls(input_tokens, output_tokens, cost)


def _gateway_cost(headers: Mapping[str, str]) -> float | None:
    """Per-call cost reported by LiteLLM (``x-litellm-response-cost``)."""
    try:
        return float(headers["x-litellm-response-cost"])
    except (KeyError, TypeError, ValueError):
        return None


class AgentError(Exception):
    """User-presentable failure; carries any spend incurred before failing."""

    def __init__(self, message: str, usage: Usage = Usage()) -> None:
        super().__init__(message)
        self.usage = usage


class PayloadError(ValueError):
    """Model output is not a usable JSON payload."""


@dataclass(frozen=True)
class QuoteResult:
    reply: str
    quote: Quote | None
    usage: Usage


@dataclass(frozen=True)
class ProvisioningResult:
    payload: dict[str, Any]
    api_response: dict[str, Any]
    attempts: int
    usage: Usage

    @property
    def succeeded(self) -> bool:
        return self.api_response.get("status") == "SUCCESS"


# --- NLU intent schema ----------------------------------------------------
_CLOUD_KEYWORDS: dict[CloudProvider, tuple[str, ...]] = {
    "AWS": ("AWS", "AMAZON"),
    "GCP": ("GCP", "GOOGLE"),
    "AZURE": ("AZURE", "MICROSOFT"),
}


class NetworkIntent(BaseModel):
    """Structured intent extracted by the NLU model (lenient by design)."""

    model_config = ConfigDict(extra="ignore")

    customer_name: str = config.DEFAULT_CUSTOMER_NAME
    source_city: str | None = None
    bandwidth_mbps: int | None = None
    cloud_provider: CloudProvider | None = None
    ddos_protection: bool = False
    monthly_egress_gb: int | None = None
    reply: str = ""

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        # JSON nulls fall back to field defaults.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data

    @field_validator("cloud_provider", mode="before")
    @classmethod
    def _normalise_cloud(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.upper()
        return next(
            (code for code, keys in _CLOUD_KEYWORDS.items()
             if any(key in text for key in keys)),
            None,
        )

    @field_validator("bandwidth_mbps", "monthly_egress_gb", mode="before")
    @classmethod
    def _non_negative(cls, value: Any) -> Any:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return round(value) if value > 0 else None
        return value


# --- OpenAI plumbing ------------------------------------------------------
_ENDPOINT_DEFAULT = "https://api.openai.com/v1"


def _endpoint() -> str:
    return config.OPENAI_BASE_URL or _ENDPOINT_DEFAULT


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        logger.error("No API key found (LITELLM_API_KEY / OPENAI_API_KEY)")
        raise AgentError(
            "No API key set. Add LITELLM_API_KEY to your .env file and restart."
        )
    if config.OPENAI_BASE_URL is None and key_shape(config.OPENAI_API_KEY) == "gateway":
        # Never send a team/gateway key to api.openai.com.
        logger.error("Gateway-style key but no LITELLM_BASE_URL configured")
        raise AgentError(
            "Your key is a LiteLLM team key, but LITELLM_BASE_URL is not set. "
            "Copy the base URL from the Hackathon LLM Access Guide into .env "
            "(LITELLM_BASE_URL=https://.../v1) and restart."
        )
    logger.info(
        "Creating OpenAI client: endpoint=%s timeout=%ss max_retries=%s key=%s",
        _endpoint(),
        config.OPENAI_TIMEOUT_SECONDS,
        config.OPENAI_MAX_RETRIES,
        describe_api_key(config.OPENAI_API_KEY),
    )
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
        max_retries=config.OPENAI_MAX_RETRIES,
    )


def _server_message(exc: openai.OpenAIError) -> str:
    """The server's own error text (OpenAI or LiteLLM), key-redacted.

    OpenAI uses ``{"error": {"message": ...}}``; LiteLLM route/permission
    errors use ``{"detail": ...}``.
    """
    body = getattr(exc, "body", None)
    text: object = None
    if isinstance(body, dict):
        inner = body.get("error")
        source = inner if isinstance(inner, dict) else body
        text = source.get("message") or source.get("detail")
    elif isinstance(body, str):
        text = body
    return redact(str(text or exc))[:500]


def _auth_help(exc: openai.OpenAIError) -> str:
    """Actionable explanation for an HTTP 401/403 from the endpoint."""
    status = getattr(exc, "status_code", 401)
    message = _server_message(exc)
    lowered = message.lower()
    if "route" in lowered:
        hint = (
            " This key may only call the routes listed above; the app uses "
            "only /chat/completions and /responses."
        )
    elif "model" in lowered:
        hint = (
            " The key cannot use this model: set NLU_MODEL / CODEX_MODEL in "
            f".env (currently {config.NLU_MODEL} / {config.CODEX_MODEL}) to "
            "one of the models named above."
        )
    else:
        hint = (
            " Check LITELLM_API_KEY (team key still active?) and that "
            "LITELLM_BASE_URL is the gateway that issued it."
        )
    return (
        f"HTTP {status} from {_endpoint()}: {message} "
        f"[key: {describe_api_key(config.OPENAI_API_KEY)}].{hint} "
        "Details: logs/broker.log."
    )


def _error_chain(exc: BaseException) -> list[str]:
    """Flatten ``__cause__``/``__context__`` into readable lines."""
    chain: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return chain


def _log_openai_error(operation: str, model: str, exc: openai.OpenAIError) -> None:
    """Log everything needed to diagnose a failed OpenAI call."""
    details: dict[str, object] = {
        "operation": operation,
        "model": model,
        "endpoint": _endpoint(),
        "error_type": type(exc).__name__,
    }
    if isinstance(exc, openai.APIStatusError):
        details |= {
            "http_status": exc.status_code,
            "request_id": exc.request_id,
            "error_body": exc.body,
            "response_text": exc.response.text[:500],
            "gateway_headers": {
                k: v
                for k, v in exc.response.headers.items()
                if k.lower().startswith(("x-litellm", "x-request-id"))
            },
        }
    logger.error("OpenAI call failed: %s", details)
    logger.error("Cause chain: %s", " <- ".join(_error_chain(exc)))
    logger.debug("Traceback for failed OpenAI call", exc_info=exc)


def _friendly_error(exc: openai.OpenAIError) -> str:
    """Translate SDK exceptions into actionable UI messages."""
    match exc:
        case openai.AuthenticationError() | openai.PermissionDeniedError():
            return _auth_help(exc)
        case openai.RateLimitError():
            return (
                f"Rate limit or budget exceeded: {_server_message(exc)} "
                "(LiteLLM team keys have per-team budgets/limits)."
            )
        case openai.APITimeoutError():
            return "OpenAI request timed out. Please retry."
        case openai.APIConnectionError() if "CERTIFICATE_VERIFY_FAILED" in " ".join(
            _error_chain(exc)
        ):
            return (
                "TLS certificate verification failed (a corporate proxy "
                "re-signing HTTPS traffic is the usual cause)."
            )
        case openai.APIConnectionError():
            return (
                f"Cannot reach {_endpoint()}. Check your network/VPN, proxy "
                "and LITELLM_BASE_URL. Details are in logs/broker.log."
            )
        case openai.NotFoundError():
            return (
                f"Not found: {_server_message(exc)} Check the model name "
                "(MODEL_PROFILE / NLU_MODEL / CODEX_MODEL) and "
                "LITELLM_BASE_URL (it usually ends in /v1)."
            )
        case openai.BadRequestError():
            return f"The endpoint rejected the request: {_server_message(exc)}"
        case openai.APIStatusError():
            return (
                f"Service error (HTTP {exc.status_code}): "
                f"{_server_message(exc)}"
            )
        case _:
            return f"Unexpected OpenAI error: {exc}"


def check_connection() -> tuple[bool, str]:
    """Live probe of both configured models via the routes the app uses.

    Team keys are route-restricted (``/v1/models`` is typically forbidden),
    so this sends two tiny requests (a few tokens each) instead of listing
    models: one to /chat/completions (NLU) and one to /responses (payload).
    """
    started = time.perf_counter()
    try:
        client = _client()
        chat = client.chat.completions.create(
            model=config.NLU_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            reasoning_effort=config.NLU_REASONING_EFFORT,
            max_completion_tokens=32,
        )
        logger.info("Probe chat.completions/%s OK: finish=%s usage=%s",
                    config.NLU_MODEL,
                    chat.choices[0].finish_reason if chat.choices else None,
                    chat.usage)
    except AgentError as exc:
        return False, str(exc)
    except openai.OpenAIError as exc:
        _log_openai_error("probe chat.completions", config.NLU_MODEL, exc)
        return False, (
            f"/chat/completions with {config.NLU_MODEL} failed: "
            f"{_friendly_error(exc)}"
        )

    try:
        resp = client.responses.create(
            model=config.CODEX_MODEL,
            input="ping",
            reasoning={"effort": config.CODEX_REASONING_EFFORT},
            max_output_tokens=16,
            store=False,
        )
        logger.info("Probe responses/%s OK: status=%s usage=%s",
                    config.CODEX_MODEL, getattr(resp, "status", None),
                    resp.usage)
    except openai.OpenAIError as exc:
        _log_openai_error("probe responses", config.CODEX_MODEL, exc)
        return False, (
            f"/responses with {config.CODEX_MODEL} failed: "
            f"{_friendly_error(exc)}"
        )

    elapsed = time.perf_counter() - started
    logger.info("Connection check OK in %.2fs", elapsed)
    return True, (
        f"Connected to {_endpoint()} in {elapsed:.1f}s: "
        f"{config.NLU_MODEL} (chat) and {config.CODEX_MODEL} (responses) "
        "both accepted."
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first {...} block, tolerating markdown fences or prose."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise PayloadError("Output contained no JSON object.")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise PayloadError(f"Output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PayloadError("Top-level JSON value must be an object.")
    return data


def _to_param(message: ChatMessage) -> ChatCompletionMessageParam:
    if message["role"] == "user":
        return {"role": "user", "content": message["content"]}
    return {"role": "assistant", "content": message["content"]}


def _call_nlu(system_prompt: str, history: Sequence[ChatMessage]) -> tuple[str, Usage]:
    """NLU model (``config.NLU_MODEL``) via /v1/chat/completions, JSON mode."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        *map(_to_param, history),
    ]
    logger.info(
        "NLU request: model=%s effort=%s messages=%d chars=%d",
        config.NLU_MODEL,
        config.NLU_REASONING_EFFORT,
        len(messages),
        sum(len(str(m["content"])) for m in messages),
    )
    logger.debug("NLU last user message: %.500s", history[-1]["content"])
    started = time.perf_counter()
    try:
        raw = _client().chat.completions.with_raw_response.create(
            model=config.NLU_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            reasoning_effort=config.NLU_REASONING_EFFORT,
            max_completion_tokens=config.NLU_MAX_COMPLETION_TOKENS,
        )
        completion = raw.parse()
    except openai.OpenAIError as exc:
        _log_openai_error("chat.completions", config.NLU_MODEL, exc)
        raise AgentError(_friendly_error(exc)) from exc

    tokens = completion.usage
    usage = Usage.for_model(
        config.NLU_MODEL,
        tokens.prompt_tokens if tokens else 0,
        tokens.completion_tokens if tokens else 0,
        _gateway_cost(raw.headers),
    )
    choice = completion.choices[0] if completion.choices else None
    logger.info(
        "NLU response: %.2fs request_id=%s finish=%s in=%d out=%d cost=$%.6f",
        time.perf_counter() - started,
        getattr(completion, "_request_id", None),
        choice.finish_reason if choice else "no_choices",
        usage.input_tokens,
        usage.output_tokens,
        usage.cost_usd,
    )
    if choice is None or not choice.message.content:
        reason = choice.finish_reason if choice else "no_choices"
        raise AgentError(f"The NLU model returned no content ({reason}).", usage)
    logger.debug("NLU raw output: %.1000s", choice.message.content)
    return choice.message.content, usage


def _call_codex(prompt: str) -> tuple[str, Usage]:
    """Payload model (``config.CODEX_MODEL``) via /v1/responses, store=False."""
    logger.info(
        "Codex request: model=%s effort=%s prompt_chars=%d",
        config.CODEX_MODEL,
        config.CODEX_REASONING_EFFORT,
        len(prompt),
    )
    started = time.perf_counter()
    try:
        raw = _client().responses.with_raw_response.create(
            model=config.CODEX_MODEL,
            instructions=_CODEX_INSTRUCTIONS,
            input=prompt,
            reasoning={"effort": config.CODEX_REASONING_EFFORT},
            max_output_tokens=config.CODEX_MAX_OUTPUT_TOKENS,
            store=False,
        )
        response = raw.parse()
    except openai.OpenAIError as exc:
        _log_openai_error("responses", config.CODEX_MODEL, exc)
        raise AgentError(_friendly_error(exc)) from exc

    tokens = response.usage
    usage = Usage.for_model(
        config.CODEX_MODEL,
        tokens.input_tokens if tokens else 0,
        tokens.output_tokens if tokens else 0,
        _gateway_cost(raw.headers),
    )
    logger.info(
        "Codex response: %.2fs request_id=%s status=%s in=%d out=%d "
        "cost=$%.6f",
        time.perf_counter() - started,
        getattr(response, "_request_id", None),
        getattr(response, "status", None),
        usage.input_tokens,
        usage.output_tokens,
        usage.cost_usd,
    )
    logger.debug("Codex raw output: %.2000s", response.output_text)
    return response.output_text, usage


# --- Quote generation (NLU model) -----------------------------------------
def _nlu_system_prompt(ctx: LumenContext) -> str:
    sites = "; ".join(
        f"{loc.city} ({', '.join((loc.site_code, *loc.aliases))})"
        for loc in ctx.locations
    )
    return (
        "You are the intake engine of Lumen's Zero-Touch NaaS Broker. Read "
        "the conversation and extract the customer's CURRENT cloud "
        "connectivity order. Later messages override earlier ones.\n\n"
        f"Lumen on-net locations: {sites}.\n"
        "Supported clouds: AWS, GCP, AZURE. Port speeds: 1G, 10G.\n\n"
        "Respond with ONLY a JSON object containing exactly these keys:\n"
        '- "customer_name": company name, or null\n'
        '- "source_city": canonical on-net city name when the user\'s '
        "location matches one (or its alias), otherwise the city as "
        "written, or null\n"
        '- "bandwidth_mbps": integer Mbps (1 Gbps = 1000), or null\n'
        '- "cloud_provider": "AWS", "GCP", "AZURE", or null when no specific '
        "cloud was named\n"
        '- "ddos_protection": true or false\n'
        '- "monthly_egress_gb": integer GB per month if stated, else null\n'
        '- "reply": friendly reply of at most 40 words that acknowledges the '
        "request or asks for a missing source city or bandwidth. Never "
        "mention prices or totals."
    )


def _quote_summary(lead: str, quote: Quote, capex_note: str) -> str:
    optimizer_note = ""
    if quote.cloud_recommended:
        optimizer_note = (
            f"💡 No cloud was specified, so the multi-cloud optimizer chose "
            f"**{quote.cloud_provider}** as the cheapest egress path"
        )
        if len(quote.cloud_options) > 1:
            runner_up = quote.cloud_options[1]
            saving = (
                runner_up.effective_hourly_usd
                - quote.cloud_options[0].effective_hourly_usd
            )
            optimizer_note += (
                f" (saves ${saving:.4f}/hr vs {runner_up.provider})"
            )
        optimizer_note += ".\n\n"

    return (
        f"{lead}\n\n"
        f"**Quote `{quote.quote_id}`** — **${quote.total_hourly_usd:,.4f}/hr** "
        f"(≈ ${quote.estimated_monthly_usd:,.2f}/mo) for "
        f"{quote.port_quantity} x {quote.port_speed} from {quote.city} to "
        f"{quote.cloud_product} ({quote.cloud_region}).\n\n"
        f"{optimizer_note}✅ {capex_note}\n\n"
        f"Review the Bill of Materials and topology, then click "
        f"**Approve Quote** (or type `{config.APPROVAL_TOKEN}`) to provision."
    ).strip()


def _build_quote_reply(
    ctx: LumenContext, intent: NetworkIntent
) -> tuple[str, Quote | None]:
    """Turn an intent into either a clarification or a priced quote."""
    on_net = ", ".join(loc.city for loc in ctx.locations)
    if intent.source_city is None:
        return (
            intent.reply
            or f"Which Lumen on-net location should we connect? ({on_net})",
            None,
        )

    location = find_location(ctx, intent.source_city)
    if location is None:
        return (
            f"**{intent.source_city}** is not a Lumen on-net location yet. "
            f"Available on-net sites: {on_net}. Which one should I use?",
            None,
        )
    if intent.bandwidth_mbps is None:
        return (
            intent.reply
            or f"How much bandwidth do you need from {location.city}?",
            None,
        )

    try:
        speed, quantity = select_port(ctx, intent.bandwidth_mbps)
    except QuoteError as exc:
        return str(exc), None
    availability = check_port_availability(location, speed, quantity)
    if not availability.available:
        alternatives = find_alternative_sites(ctx, location, speed, quantity)
        suggestion = (
            " Nearby on-net sites with capacity: "
            + ", ".join(f"{a.city} ({a.site_code})" for a in alternatives)
            + "."
            if alternatives
            else " Please try a lower bandwidth."
        )
        return f"⚠️ {availability.message}{suggestion}", None

    try:
        quote = calculate_quote(
            ctx,
            customer_name=intent.customer_name,
            location=location,
            bandwidth_mbps=intent.bandwidth_mbps,
            speed=speed,
            port_quantity=quantity,
            cloud_provider=intent.cloud_provider,
            ddos_protection=intent.ddos_protection,
            monthly_egress_gb=(
                intent.monthly_egress_gb
                if intent.monthly_egress_gb is not None
                else ctx.default_monthly_egress_gb
            ),
        )
    except QuoteError as exc:
        return str(exc), None
    # The model's free-text reply is only used for clarifications; a priced
    # quote gets a deterministic lead so it can never contradict the quote.
    lead = f"Thanks, {quote.customer_name} — here is your quote."
    return _quote_summary(lead, quote, availability.message), quote


def generate_quote(
    history: Sequence[ChatMessage], ctx: LumenContext
) -> QuoteResult:
    """Extract intent with the NLU model and price the Bill of Materials."""
    window = list(history)[-config.NLU_HISTORY_WINDOW:]
    if not any(m["role"] == "user" for m in window):
        raise AgentError("Please describe the network connection you need.")

    raw, usage = _call_nlu(_nlu_system_prompt(ctx), window)
    try:
        intent = NetworkIntent.model_validate(_extract_json_object(raw))
    except ValueError as exc:  # PayloadError or pydantic ValidationError
        logger.warning("Unparseable NLU output %r: %s", raw, exc)
        raise AgentError(
            "I couldn't interpret that request. Please restate the source "
            "city, bandwidth and (optionally) the cloud provider.",
            usage,
        ) from exc

    logger.info("Extracted intent: %s", intent.model_dump())
    reply, quote = _build_quote_reply(ctx, intent)
    return QuoteResult(reply=reply, quote=quote, usage=usage)


# --- Topology rendering ($0, deterministic) -------------------------------
def _label(text: str) -> str:
    """Strip characters that would break Mermaid node/edge labels."""
    return re.sub(r'["<>{}|\[\]()#;]', "", text)


def generate_mermaid_diagram(quote: Quote) -> str:
    """Mermaid flowchart: Source -> Lumen PoP -> NaaS Fabric -> Cloud."""
    port = _label(f"{quote.port_quantity} x {quote.port_speed}")
    lines = [
        "flowchart LR",
        f'    SRC["Customer Site<br/>{_label(quote.customer_name)}<br/>'
        f'{_label(quote.city)}, {quote.country_code}"]',
        f'    POP["Lumen PoP<br/>{_label(quote.pop_name)}<br/>'
        f'{quote.site_code}"]',
        '    FABRIC(("Lumen NaaS<br/>Fabric"))',
        f'    CLOUD["{_label(quote.cloud_product)}<br/>'
        f'{_label(quote.cloud_region)}"]',
        f'    SRC -->|"{port} access"| POP',
        '    POP --> FABRIC',
        f'    FABRIC -->|"{quote.cloud_provider} on-ramp"| CLOUD',
        "    classDef lumen fill:#0047bb,stroke:#002a6e,color:#ffffff",
        "    classDef cloud fill:#fff4e5,stroke:#ff9900,color:#222222",
        "    class POP,FABRIC lumen",
        "    class CLOUD cloud",
    ]
    if quote.ddos_protection:
        lines += [
            '    DDOS{{"DDoS Mitigation"}} -.->|"scrubbing"| POP',
            "    classDef shield fill:#e8f7ee,stroke:#1a7f37,color:#1a7f37",
            "    class DDOS shield",
        ]
    return "\n".join(lines)


# --- Provisioning (payload model + self-healing) ---------------------------
_CODEX_INSTRUCTIONS = (
    "You are a deterministic provisioning-payload compiler for the Lumen "
    "ordering API. Output exactly one JSON object that conforms to the given "
    "JSON Schema and the field mapping. No markdown, no comments, no prose."
)

# JSON path in the payload -> Quote attribute; also drives reconciliation.
_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("quoteId", "quote_id"),
    ("customer.name", "customer_name"),
    ("serviceLocation.siteCode", "site_code"),
    ("serviceLocation.city", "city"),
    ("serviceLocation.countryCode", "country_code"),
    ("port.speed", "port_speed"),
    ("port.quantity", "port_quantity"),
    ("port.bandwidthMbps", "bandwidth_mbps"),
    ("cloudConnection.provider", "cloud_provider"),
    ("cloudConnection.region", "cloud_region"),
    ("cloudConnection.product", "cloud_product"),
    ("addOns.ddosProtection", "ddos_protection"),
    ("billing.hourlyRateUsd", "total_hourly_usd"),
    ("billing.estimatedMonthlyUsd", "estimated_monthly_usd"),
)
_CONSTANTS: dict[str, object] = {
    "orderType": "NEW_INSTALL",
    "productFamily": "NAAS_CLOUD_CONNECT",
    "billing.pricingModel": "HOURLY",
    "billing.currency": "USD",
}
_PRICE_TOLERANCE_USD = 0.005


def is_quote_approved(history: Sequence[ChatMessage]) -> bool:
    """True only if the latest user message carries the approval token."""
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    return config.APPROVAL_TOKEN in last_user.upper()


def _expected_fields(quote: Quote) -> dict[str, object]:
    return {path: getattr(quote, attr) for path, attr in _FIELD_MAP} | _CONSTANTS


def _dig(payload: dict[str, Any], path: str) -> Any:
    node: Any = payload
    for key in path.split("."):
        node = node.get(key) if isinstance(node, dict) else None
    return node


def _reconcile(payload: dict[str, Any], quote: Quote) -> list[str]:
    """Business-level check: every payload value must match the quote."""

    def matches(actual: Any, expected: object) -> bool:
        if isinstance(expected, float) and isinstance(actual, int | float):
            return math.isclose(actual, expected, abs_tol=_PRICE_TOLERANCE_USD)
        return bool(actual == expected)

    return [
        f"- {path}: expected {expected!r}, got {_dig(payload, path)!r}"
        for path, expected in _expected_fields(quote).items()
        if not matches(_dig(payload, path), expected)
    ]


def _codex_prompt(ctx: LumenContext, quote: Quote, feedback: str) -> str:
    api = ctx.order_request_api
    mapping = "\n".join(
        [f"- {path} = quote.{attr}" for path, attr in _FIELD_MAP]
        + [f"- {path} = {json.dumps(value)}" for path, value in _CONSTANTS.items()]
    )
    sections = [
        f"Target endpoint: {api.method} {api.path}",
        "JSON Schema:\n" + json.dumps(api.request_schema, indent=2),
        "Approved quote (source of truth, copy values exactly):\n"
        + quote.model_dump_json(indent=2, exclude={"line_items", "cloud_options"}),
        "Field mapping:\n" + mapping,
        feedback,
    ]
    return "\n\n".join(s for s in sections if s)


def _validate_payload(raw: str, quote: Quote) -> dict[str, Any]:
    """Schema validation (Pydantic) followed by quote reconciliation."""
    order = OrderRequest.model_validate(_extract_json_object(raw))
    payload = order.model_dump(mode="json", by_alias=True)
    mismatches = _reconcile(payload, quote)
    if mismatches:
        raise PayloadError(
            "Payload does not match the approved quote:\n" + "\n".join(mismatches)
        )
    return payload


def generate_provisioning_payload(
    history: Sequence[ChatMessage], quote: Quote, ctx: LumenContext
) -> ProvisioningResult:
    """Compile the approved quote with Codex, self-heal, then mock-provision.

    Requires the latest user message to contain ``QUOTE_APPROVED``. Each
    rejected payload is fed back to Codex with the exact validation errors,
    up to ``MAX_HEAL_ATTEMPTS`` retries.
    """
    if not is_quote_approved(history):
        logger.warning("Provisioning blocked: %s not in latest user message",
                       config.APPROVAL_TOKEN)
        raise AgentError(
            f"Provisioning requires explicit approval ({config.APPROVAL_TOKEN})."
        )

    logger.info("Provisioning started for quote %s", quote.quote_id)
    usage = Usage()
    feedback = ""
    errors = ""
    max_attempts = config.MAX_HEAL_ATTEMPTS + 1
    for attempt in range(1, max_attempts + 1):
        logger.info("Codex attempt %d/%d", attempt, max_attempts)
        try:
            raw, call_usage = _call_codex(_codex_prompt(ctx, quote, feedback))
        except AgentError as exc:
            raise AgentError(str(exc), usage + exc.usage) from exc
        usage = usage + call_usage

        try:
            payload = _validate_payload(raw, quote)
        except ValidationError as exc:
            errors = format_validation_errors(exc)
        except PayloadError as exc:
            errors = str(exc)
        else:
            logger.info("Payload valid on attempt %d; calling mock API", attempt)
            api_response = mock_lumen_provisioning_api(payload)
            return ProvisioningResult(payload, api_response, attempt, usage)

        logger.warning("Codex attempt %d rejected:\n%s", attempt, errors)
        feedback = (
            "Your previous output was REJECTED by validation:\n"
            f"{errors}\n\nPrevious output:\n{raw[:4000]}\n\n"
            "Return the corrected JSON object only."
        )

    raise AgentError(
        f"Codex could not produce a valid payload after {max_attempts} "
        f"attempts. Last validation errors:\n{errors}",
        usage,
    )
