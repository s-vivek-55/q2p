"""Central configuration: environment variables, model routing and constants."""

import os
from pathlib import Path
from typing import Final, Literal, NamedTuple

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
CONTEXT_PATH: Final[Path] = BASE_DIR / "lumen_context.json"

# --- OpenAI-compatible endpoint (hackathon LiteLLM gateway) ---------------
# LITELLM_* names match the Hackathon LLM Access Guide; OPENAI_* still work.
KEY_SOURCE: Final[str] = next(
    (n for n in ("LITELLM_API_KEY", "OPENAI_API_KEY") if os.getenv(n)), "<none>"
)
BASE_URL_SOURCE: Final[str] = next(
    (n for n in ("LITELLM_BASE_URL", "OPENAI_BASE_URL") if os.getenv(n)),
    "<none>",
)
OPENAI_API_KEY: Final[str | None] = (
    os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None
)
OPENAI_BASE_URL: Final[str | None] = (
    (os.getenv("LITELLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "")
    .strip()
    .rstrip("/")
    or None
)
OPENAI_TIMEOUT_SECONDS: Final[float] = float(
    os.getenv("OPENAI_TIMEOUT_SECONDS", "60")
)
OPENAI_MAX_RETRIES: Final[int] = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

# --- Model routing (cost-aware) -----------------------------------------
ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class ModelProfile(NamedTuple):
    nlu_model: str
    nlu_effort: ReasoningEffort
    codex_model: str
    codex_effort: ReasoningEffort


# "dev" (default) runs everything on the cheapest model that works on the
# hackathon gateway: gpt-5.6-luna was measured at ~40% of gpt-5-mini's cost
# per call on both endpoints, with valid output every time. Set
# MODEL_PROFILE=prod for the target production routing.
MODEL_PROFILES: Final[dict[str, ModelProfile]] = {
    "dev": ModelProfile("gpt-5.6-luna", "low", "gpt-5.6-luna", "low"),
    "prod": ModelProfile("gpt-5-mini", "low", "gpt-5.3-codex", "low"),
}
MODEL_PROFILE: Final[str] = os.getenv("MODEL_PROFILE", "dev").strip().lower()
if MODEL_PROFILE not in MODEL_PROFILES:
    raise ValueError(
        f"MODEL_PROFILE must be one of {sorted(MODEL_PROFILES)}, "
        f"got {MODEL_PROFILE!r}"
    )
_PROFILE: Final[ModelProfile] = MODEL_PROFILES[MODEL_PROFILE]

# NLU_MODEL / CODEX_MODEL env vars override the profile's model names.
NLU_MODEL: Final[str] = os.getenv("NLU_MODEL") or _PROFILE.nlu_model
CODEX_MODEL: Final[str] = os.getenv("CODEX_MODEL") or _PROFILE.codex_model
NLU_REASONING_EFFORT: Final[ReasoningEffort] = _PROFILE.nlu_effort
CODEX_REASONING_EFFORT: Final[ReasoningEffort] = _PROFILE.codex_effort
NLU_MAX_COMPLETION_TOKENS: Final[int] = 2_000
CODEX_MAX_OUTPUT_TOKENS: Final[int] = 4_000
NLU_HISTORY_WINDOW: Final[int] = 12  # last N chat messages sent to the NLU
MAX_HEAL_ATTEMPTS: Final[int] = 2  # Codex retries after a rejected payload

# --- Budget guard --------------------------------------------------------
# USD per 1M tokens as (input, output). Used only for the in-app spend meter;
# adjust if OpenAI pricing changes.
MODEL_PRICING_PER_MTOK: Final[dict[str, tuple[float, float]]] = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.3-codex": (1.75, 14.00),
}
SESSION_BUDGET_USD: Final[float] = float(
    os.getenv("SESSION_BUDGET_USD", "2.00")
)

# --- Workflow ------------------------------------------------------------
APPROVAL_TOKEN: Final[str] = "QUOTE_APPROVED"
MOCK_API_LATENCY_SECONDS: Final[float] = 2.0
DEFAULT_CUSTOMER_NAME: Final[str] = "Enterprise Customer"
