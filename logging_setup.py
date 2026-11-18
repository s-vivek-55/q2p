"""Application logging: console + rotating file, with API-key redaction.

``setup_logging()`` is idempotent (Streamlit re-executes the script on every
interaction). Detailed DEBUG output goes to ``logs/broker.log``; the console
shows INFO and above.
"""

import hashlib
import logging
import os
import platform
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

import config

LOG_DIR: Final[Path] = config.BASE_DIR / "logs"
LOG_FILE: Final[Path] = LOG_DIR / "broker.log"
_HANDLER_NAME: Final[str] = "broker-file"
_LOG_FORMAT: Final[str] = (
    "%(asctime)s %(levelname)-7s %(name)s:%(lineno)d | %(message)s"
)

# Anything that looks like an API key or bearer token is masked in all logs.
_SECRET_RE: Final = re.compile(
    r"(?P<key>sk-[A-Za-z0-9_\-]{8,})|(?P<bearer>Bearer\s+\S+)", re.IGNORECASE
)


def redact(text: str) -> str:
    """Mask API keys and bearer tokens, keeping a short prefix for triage."""

    def mask(match: re.Match[str]) -> str:
        if match.group("key"):
            return f"{match.group('key')[:6]}…<redacted>"
        return "Bearer <redacted>"

    return _SECRET_RE.sub(mask, text)


class _RedactingFormatter(logging.Formatter):
    """Redacts the fully rendered record, including tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def key_shape(key: str | None) -> str:
    """Classify a key: ``openai`` (sk-proj-…), ``gateway`` (LiteLLM) or ``unknown``."""
    if not key:
        return "unknown"
    if key.startswith("sk-proj-") or re.fullmatch(r"sk-[A-Za-z0-9]{40,}", key):
        return "openai"
    if re.fullmatch(r"sk-[A-Za-z0-9_\-]{22}", key):
        return "gateway"
    return "unknown"


def describe_api_key(key: str | None) -> str:
    """Non-reversible description of a key: length, shape and fingerprint."""
    if not key:
        return "<not set>"
    labels = {
        "openai": "OpenAI-style",
        "gateway": "LiteLLM/gateway-style",
        "unknown": "unrecognised format",
    }
    fingerprint = hashlib.sha256(key.encode()).hexdigest()[:8]
    return (
        f"len={len(key)} prefix={key[:6]}… sha256={fingerprint} "
        f"format={labels[key_shape(key)]}"
    )


def setup_logging() -> None:
    """Configure root logging once per process."""
    root = logging.getLogger()
    if any(h.get_name() == _HANDLER_NAME for h in root.handlers):
        return

    LOG_DIR.mkdir(exist_ok=True)
    formatter = _RedactingFormatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.set_name(_HANDLER_NAME)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console)

    # Chatty third-party loggers: keep request lines, drop wire-level noise.
    for name in ("httpx", "httpx2", "openai"):
        logging.getLogger(name).setLevel(logging.INFO)
    for name in ("httpcore", "httpcore2", "urllib3", "watchdog", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in ("streamlit", "asyncio", "fsevents"):
        logging.getLogger(name).setLevel(logging.INFO)


def log_startup_diagnostics() -> None:
    """Record environment facts that explain most connectivity/auth issues."""
    import openai  # local import keeps this module importable without it
    import streamlit

    logger = logging.getLogger("diagnostics")
    proxy_vars = {
        name: bool(os.getenv(name))
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
                     "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
    }
    logger.info("=== Broker startup diagnostics ===")
    logger.info("python=%s (%s) on %s",
                platform.python_version(), sys.executable, platform.platform())
    logger.info("openai=%s streamlit=%s",
                openai.__version__, streamlit.__version__)
    logger.info("model_profile=%s nlu_model=%s(%s) codex_model=%s(%s)",
                config.MODEL_PROFILE, config.NLU_MODEL,
                config.NLU_REASONING_EFFORT, config.CODEX_MODEL,
                config.CODEX_REASONING_EFFORT)
    logger.info("endpoint=%s (from %s) timeout=%ss max_retries=%s",
                config.OPENAI_BASE_URL or "<default https://api.openai.com/v1>",
                config.BASE_URL_SOURCE,
                config.OPENAI_TIMEOUT_SECONDS, config.OPENAI_MAX_RETRIES)
    logger.info("api_key (from %s): %s", config.KEY_SOURCE,
                describe_api_key(config.OPENAI_API_KEY))
    logger.info("env (set?)=%s", proxy_vars)
    logger.info("log_file=%s", LOG_FILE)


def tail_log(lines: int = 80) -> str:
    """Last ``lines`` lines of the log file, for display in the UI."""
    try:
        with LOG_FILE.open(encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return "(no log file yet)"
