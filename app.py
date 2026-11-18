"""Streamlit entry point for the Lumen Zero-Touch NaaS Broker.

Run with:  streamlit run app.py   (``python app.py`` also works: it
re-launches itself under Streamlit with the same interpreter).
"""

import html
import logging
import subprocess
import sys

try:
    import streamlit as st
    import streamlit.components.v1 as components
    from streamlit import runtime

    import config
    from agent import (
        AgentError,
        ChatMessage,
        ProvisioningResult,
        Usage,
        check_connection,
        generate_mermaid_diagram,
        generate_provisioning_payload,
        generate_quote,
    )
    from logging_setup import (
        LOG_FILE,
        log_startup_diagnostics,
        setup_logging,
        tail_log,
    )
    from tools import ContextError, LumenContext, Quote, load_context
except ModuleNotFoundError as missing:
    sys.exit(
        f"Missing dependency '{missing.name}' for {sys.executable}.\n"
        f"Install with: \"{sys.executable}\" -m pip install -r requirements.txt"
    )

setup_logging()
logger = logging.getLogger("app")


@st.cache_resource
def _log_diagnostics_once() -> bool:
    """Run startup diagnostics once per server process (not per rerun)."""
    log_startup_diagnostics()
    return True


WELCOME = (
    "Hi! Describe the connectivity you need — for example: *\"10 Gbps from "
    "our Seattle office to the cloud with DDoS protection.\"* I'll price it, "
    "draw the topology and provision it as soon as you approve."
)


# --- Session state ----------------------------------------------------------
def _init_state() -> None:
    defaults: dict[str, object] = {
        "messages": [ChatMessage(role="assistant", content=WELCOME)],
        "history_start": 0,  # index where the current quote conversation begins
        "quote": None,
        "provisioning": None,
        "spent_usd": 0.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_state() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def _messages() -> list[ChatMessage]:
    messages: list[ChatMessage] = st.session_state.messages
    return messages


def _say(role: str, content: str) -> None:
    _messages().append(
        ChatMessage(role="user" if role == "user" else "assistant", content=content)
    )


def _record_usage(usage: Usage) -> None:
    st.session_state.spent_usd += usage.cost_usd


def _within_budget() -> bool:
    if st.session_state.spent_usd < config.SESSION_BUDGET_USD:
        return True
    _say(
        "assistant",
        f"🛑 Session budget of ${config.SESSION_BUDGET_USD:.2f} reached. "
        "Start a new session or raise SESSION_BUDGET_USD.",
    )
    return False


# --- Workflow handlers ----------------------------------------------------
def _handle_user_prompt(prompt: str, ctx: LumenContext) -> None:
    """Chat turn: either an approval or a new/updated quote request."""
    if config.APPROVAL_TOKEN in prompt.upper():
        _handle_approval(ctx, from_chat=prompt)
        return

    _say("user", prompt)
    logger.info("User prompt (%d chars): %.200s", len(prompt), prompt)
    if not _within_budget():
        return
    history = _messages()[st.session_state.history_start:]
    try:
        result = generate_quote(history, ctx)
    except AgentError as exc:
        logger.warning("generate_quote failed: %s", exc)
        _record_usage(exc.usage)
        _say("assistant", f"⚠️ {exc}")
        return
    except Exception:  # noqa: BLE001 - last-resort guard; full trace is logged
        logger.exception("Unexpected error in generate_quote")
        _say("assistant", "⚠️ Unexpected error. See logs/broker.log for details.")
        return

    _record_usage(result.usage)
    logger.info("Quote turn done: quote=%s spent_total=$%.6f",
                result.quote.quote_id if result.quote else None,
                st.session_state.spent_usd)
    _say("assistant", result.reply)
    if result.quote is not None:
        st.session_state.quote = result.quote
        st.session_state.provisioning = None


def _handle_approval(ctx: LumenContext, from_chat: str | None = None) -> None:
    """Approve the active quote, compile it with Codex and mock-provision."""
    _say("user", from_chat or config.APPROVAL_TOKEN)
    quote: Quote | None = st.session_state.quote
    logger.info("Approval requested (open quote=%s)",
                quote.quote_id if quote else None)
    if quote is None:
        _say("assistant", "There is no open quote to approve yet.")
        return
    if st.session_state.provisioning is not None:
        _say("assistant", f"Quote `{quote.quote_id}` is already provisioned.")
        return
    if not _within_budget():
        return

    try:
        result = generate_provisioning_payload(
            _messages()[st.session_state.history_start:], quote, ctx
        )
    except AgentError as exc:
        logger.warning("generate_provisioning_payload failed: %s", exc)
        _record_usage(exc.usage)
        _say("assistant", f"⚠️ Provisioning failed: {exc}")
        return
    except Exception:  # noqa: BLE001 - last-resort guard; full trace is logged
        logger.exception("Unexpected error in generate_provisioning_payload")
        _say("assistant", "⚠️ Unexpected error. See logs/broker.log for details.")
        return

    _record_usage(result.usage)
    st.session_state.provisioning = result
    if result.succeeded:
        healed = (
            f" (self-healed after {result.attempts - 1} rejected draft(s))"
            if result.attempts > 1
            else ""
        )
        _say(
            "assistant",
            f"🎉 **{result.api_response['message']}** — transaction "
            f"`{result.api_response['transaction_id']}` for quote "
            f"`{quote.quote_id}`{healed}. Billing starts at "
            f"${quote.total_hourly_usd:,.4f}/hr.",
        )
        # The next request starts a fresh quote conversation.
        st.session_state.history_start = len(_messages())
    else:
        _say(
            "assistant",
            f"⚠️ Mock Lumen API rejected the order: "
            f"{result.api_response.get('message', 'unknown error')}",
        )


# --- Rendering ------------------------------------------------------------
def render_mermaid(code: str, height: int = 340) -> None:
    """Render Mermaid in an iframe (Streamlit has no native Mermaid support).

    Labels are sanitised in ``generate_mermaid_diagram`` and HTML-escaped
    here, so no user/LLM text can inject markup into the iframe.
    """
    page = f"""
        <div class="mermaid" style="display:flex;justify-content:center;">
        {html.escape(code)}
        </div>
        <script type="module">
          import mermaid from
            "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
          mermaid.initialize({{ startOnLoad: false, securityLevel: "strict",
                                theme: "neutral" }});
          await mermaid.run({{ querySelector: ".mermaid" }});
        </script>
        """
    if hasattr(st, "iframe"):  # Streamlit >= 1.5x replacement API
        st.iframe(page, height=height)
    else:
        components.html(page, height=height)


def _render_sidebar(ctx: LumenContext) -> None:
    with st.sidebar:
        st.header("⚙️ Broker Console")
        if not config.OPENAI_API_KEY:
            st.warning("LITELLM_API_KEY is not set — LLM calls will fail.")
        elif config.OPENAI_BASE_URL is None:
            st.warning(
                "LITELLM_BASE_URL is not set. Paste the base URL from the "
                "Hackathon LLM Access Guide into .env and restart."
            )
        st.markdown(
            f"**Model profile:** `{config.MODEL_PROFILE}`  \n"
            f"**NLU / BOM:** `{config.NLU_MODEL}` · chat.completions  \n"
            f"**Payload compiler:** `{config.CODEX_MODEL}` · responses  \n"
            f"**Endpoint:** `{config.OPENAI_BASE_URL or 'api.openai.com'}`  \n"
            "**Provisioning:** local mock API (no live calls)"
        )
        if st.button("🔌 Test OpenAI connection"):
            with st.spinner("Contacting the endpoint…"):
                ok, message = check_connection()
            (st.success if ok else st.error)(message)
        spent = float(st.session_state.spent_usd)
        st.progress(
            min(spent / config.SESSION_BUDGET_USD, 1.0),
            text=f"Est. spend ${spent:.4f} / ${config.SESSION_BUDGET_USD:.2f}",
        )
        with st.expander("🪵 Diagnostic log"):
            st.caption(f"Full DEBUG log: `{LOG_FILE}`")
            st.code(tail_log(60), language="log", wrap_lines=True)
        with st.expander("Lumen on-net locations"):
            st.dataframe(
                [
                    {
                        "Site": loc.site_code,
                        "City": loc.city,
                        "Geo": loc.geo,
                        "Idle 1G": loc.available_ports.get("1G", 0),
                        "Idle 10G": loc.available_ports.get("10G", 0),
                    }
                    for loc in ctx.locations
                ],
                hide_index=True,
            )
        if st.button("🔄 New session"):
            _reset_state()
            st.rerun()


def _render_quote(quote: Quote) -> None:
    st.subheader(f"Quote `{quote.quote_id}`")
    hourly, monthly, capex = st.columns(3)
    hourly.metric("Hourly", f"${quote.total_hourly_usd:,.4f}")
    monthly.metric("Est. monthly", f"${quote.estimated_monthly_usd:,.2f}")
    capex.metric("CapEx", "$0.00")

    st.markdown("**Bill of Materials**")
    st.dataframe(
        [
            {
                "SKU": item.sku,
                "Description": item.description,
                "Qty": item.quantity,
                "Unit $/hr": f"{item.unit_hourly_usd:.4f}",
                "Line $/hr": f"{item.hourly_usd:.4f}",
            }
            for item in quote.line_items
        ],
        hide_index=True,
    )

    if quote.cloud_recommended:
        st.info(
            f"💡 Multi-cloud optimizer selected **{quote.cloud_provider}** "
            f"({quote.cloud_region}) as the cheapest egress path."
        )
    with st.expander("Multi-cloud cost comparison"):
        st.dataframe(
            [
                {
                    "Cloud": opt.provider,
                    "Region": opt.region,
                    "On-ramp $/hr": f"{opt.onramp_hourly_usd:.4f}",
                    "Egress $/hr": f"{opt.egress_hourly_usd:.4f}",
                    "Effective $/hr": f"{opt.effective_hourly_usd:.4f}",
                }
                for opt in quote.cloud_options
            ],
            hide_index=True,
        )

    st.markdown("**Proposed topology**")
    diagram = generate_mermaid_diagram(quote)
    render_mermaid(diagram)
    with st.expander("Mermaid source"):
        st.code(diagram, language="text")


def _render_provisioning(result: ProvisioningResult) -> None:
    if result.succeeded:
        st.success(
            f"✅ {result.api_response['message']} — "
            f"`{result.api_response['transaction_id']}`"
        )
    else:
        st.error(f"❌ {result.api_response.get('message', 'Order rejected')}")
    with st.expander("Validated orderRequest payload", expanded=False):
        st.json(result.payload)
    with st.expander("Mock Lumen API response", expanded=False):
        st.json(result.api_response)


# --- Main -----------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Lumen Zero-Touch NaaS Broker", page_icon="🌐", layout="wide"
    )
    _init_state()
    _log_diagnostics_once()

    try:
        ctx = load_context()
    except ContextError as exc:
        st.error(f"Unable to start: {exc}")
        st.stop()

    _render_sidebar(ctx)
    st.title("🌐 Lumen Zero-Touch NaaS Broker")
    st.caption(
        "Natural language → priced Bill of Materials → validated order → "
        "provisioned network. Mock environment: no live Lumen APIs are called."
    )

    prompt = st.chat_input("Describe the network you need…")
    chat_col, quote_col = st.columns([3, 2], gap="large")

    with chat_col:
        for message in _messages():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"), st.spinner("Working on it…"):
                _handle_user_prompt(prompt, ctx)
            st.rerun()

    with quote_col:
        quote: Quote | None = st.session_state.quote
        provisioning: ProvisioningResult | None = st.session_state.provisioning
        if quote is None:
            st.info("Your quote, BOM and network topology will appear here.")
            return

        _render_quote(quote)
        if provisioning is not None:
            _render_provisioning(provisioning)
        elif st.button("✅ Approve Quote", type="primary"):
            with st.spinner(
                f"{config.CODEX_MODEL} is compiling the order and calling the "
                "mock Lumen API…"
            ):
                _handle_approval(ctx)
            st.rerun()


if runtime.exists():
    main()
elif __name__ == "__main__":
    # Plain `python app.py`: hand over to the Streamlit server.
    sys.exit(
        subprocess.call(
                [sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]
        )
    )
