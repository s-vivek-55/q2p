"""Streamlit page for the new-customer onboarding flow.

The page reuses the existing quote and provisioning renderers from ``app.py``
(passed in as callbacks), so a new customer's chosen plan gets the same BOM,
cloud comparison, topology diagram and Codex provisioning as an existing one.
"""

import logging
from collections.abc import Callable

import streamlit as st
from pydantic import ValidationError

import config
from agent import (
    AgentError,
    ChatMessage,
    ProvisioningResult,
    generate_provisioning_payload,
)
from mock_api import format_validation_errors
from onboarding import (
    NewCustomerIntent,
    NewCustomerRequirement,
    PlanOption,
    Proposal,
    build_proposal,
    create_account,
    extract_requirement,
)
from tools import LumenContext, Quote, QuoteError, find_location

logger = logging.getLogger(__name__)

EXAMPLE = (
    "I need a new connection for my office in Seattle. We are Contoso: 5 "
    "floors, about 50,000 sq ft, 10 Gbps, with DDoS protection. Contact: "
    "Jane Doe, jane@contoso.com"
)
_CLOUDS: tuple[str, ...] = ("Compare all", "AWS", "GCP", "AZURE")
_STATE_DEFAULTS: dict[str, object] = {
    "nc_intent": None,
    "nc_proposal": None,
    "nc_account": None,
    "nc_provisioning": None,
    "nc_ordered_quote": None,
    "nc_error": "",
}


def _budget_ok() -> bool:
    if st.session_state.spent_usd < config.SESSION_BUDGET_USD:
        return True
    st.error(f"🛑 Session budget of ${config.SESSION_BUDGET_USD:.2f} reached.")
    return False


def _clamp(value: int | None, low: int, high: int, default: int) -> int:
    """Keep LLM-extracted values inside the widget's allowed range."""
    return default if value is None else max(low, min(value, high))


# --- Intake ------------------------------------------------------------------
def _autofill(text: str, ctx: LumenContext) -> None:
    if not _budget_ok():
        return
    try:
        with st.spinner(f"{config.NLU_MODEL} is reading your requirement…"):
            intent, usage = extract_requirement(text, ctx)
    except AgentError as exc:
        st.session_state.spent_usd += exc.usage.cost_usd
        st.error(f"⚠️ {exc} You can still fill in the form below.")
        return
    st.session_state.spent_usd += usage.cost_usd
    st.session_state.nc_intent = intent
    if intent.reply:
        st.info(intent.reply)


def _intake_form(ctx: LumenContext) -> NewCustomerRequirement | None:
    """Editable form (prefilled by the NLU); returns a valid requirement."""
    intent: NewCustomerIntent = st.session_state.nc_intent or NewCustomerIntent()
    cities = [loc.city for loc in ctx.locations]
    match = find_location(ctx, intent.source_city)
    if intent.source_city and match is None:
        st.warning(f"{intent.source_city} is not on-net yet; pick the nearest "
                   "Lumen site below.")
    company_default = ("" if intent.customer_name == config.DEFAULT_CUSTOMER_NAME
                       else intent.customer_name)
    egress_default = ctx.default_monthly_egress_gb

    with st.form("nc_form"):
        company = st.text_input("Company", company_default)
        c1, c2 = st.columns(2)
        contact = c1.text_input("Contact name", intent.contact_name or "")
        email = c2.text_input("Contact email", intent.contact_email or "")
        city = st.selectbox("Office location (Lumen on-net)", cities,
                            index=cities.index(match.city) if match else 0)
        c3, c4 = st.columns(2)
        floors = c3.number_input(
            "Floors", 1, 200, _clamp(intent.floors, 1, 200, 1))
        area = c4.number_input(
            "Area (sq ft, 0 = unknown)", 0, 10_000_000,
            _clamp(intent.area_sqft, 0, 10_000_000, 0), step=1000)
        c5, c6 = st.columns(2)
        bandwidth = c5.number_input(
            "Required speed (Mbps, 0 = size it for me)", 0, 1_000_000,
            _clamp(intent.bandwidth_mbps, 0, 1_000_000, 0), step=1000)
        egress = c6.number_input(
            "Cloud egress (GB/month)", 0, 10_000_000,
            _clamp(intent.monthly_egress_gb, 0, 10_000_000, egress_default),
            step=500)
        cloud = st.selectbox(
            "Preferred cloud", _CLOUDS,
            index=_CLOUDS.index(intent.cloud_provider)
            if intent.cloud_provider else 0)
        ddos = st.checkbox("DDoS protection", intent.ddos_protection)
        if not st.form_submit_button("🔍 Check availability & compare plans",
                                     type="primary"):
            return None

    try:
        return NewCustomerRequirement.model_validate({
            "company_name": company.strip(),
            "contact_name": contact.strip(),
            "contact_email": email.strip(),
            "city": city,
            "floors": int(floors),
            "area_sqft": int(area),
            "bandwidth_mbps": int(bandwidth),
            "ddos_protection": ddos,
            "cloud_provider": None if cloud == _CLOUDS[0] else cloud,
            "monthly_egress_gb": int(egress),
        })
    except ValidationError as exc:
        st.error("Please fix the form:\n" + format_validation_errors(exc))
        return None


# --- Proposal ------------------------------------------------------------------
def _option_label(option: PlanOption) -> str:
    q = option.quote
    return (f"{option.name} · {q.cloud_provider} · {q.port_quantity} x "
            f"{q.port_speed} · ${q.estimated_monthly_usd:,.2f}/mo")


def _render_proposal(p: Proposal) -> int | None:
    """Show availability, sizing and plans; return the chosen option index."""
    loc, s = p.location, p.sizing
    st.subheader(f"📍 {loc.city} — {loc.pop_name} ({loc.site_code})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Est. users", f"{s.estimated_users:,}")
    m2.metric("Peak demand", f"{s.demand_mbps:,} Mbps")
    m3.metric("Recommended", f"{s.recommended_mbps:,} Mbps")
    m4.metric("Floor switches", s.floor_switches)
    st.markdown("\n".join(f"- {note}" for note in s.notes))

    st.markdown("**Services available at this location**")
    st.dataframe(
        [{"Service": svc["name"], "What it is": svc["description"],
          "From $/hr": f"{svc['fromHourlyUsd']:.2f}"} for svc in p.services],
        hide_index=True,
    )
    for warning in p.warnings:
        st.warning(warning)
    if not p.options or p.best is None:
        st.error("No plan can be delivered on-net here at this speed. Lower "
                 "the speed or choose one of the suggested sites.")
        return None

    st.markdown("**Price plans — AWS vs GCP vs Azure**")
    preferred = p.requirement.cloud_provider
    st.dataframe(
        [{"": ("⭐ Best" if o is p.best else "")
          + (" ♥ Preferred" if o.quote.cloud_provider == preferred else ""),
          "Plan": o.name, "Cloud": o.quote.cloud_provider,
          "Region": o.quote.cloud_region,
          "Port": f"{o.quote.port_quantity} x {o.quote.port_speed}",
          "DDoS": "✓" if o.quote.ddos_protection else "—",
          "$/hr": f"{o.quote.total_hourly_usd:.4f}",
          "$/month": f"{o.quote.estimated_monthly_usd:,.2f}"}
         for o in p.options],
        hide_index=True,
    )
    best = p.best.quote
    priciest = max(o.quote.estimated_monthly_usd for o in p.options
                   if o.name == p.best.name)
    st.success(
        f"⭐ Best optimal plan: **{p.best.name} via {best.cloud_provider}** "
        f"({best.cloud_region}) — ${best.total_hourly_usd:,.4f}/hr "
        f"(≈ ${best.estimated_monthly_usd:,.2f}/mo), saving "
        f"${priciest - best.estimated_monthly_usd:,.2f}/mo against the "
        "most expensive cloud for the same plan. $0 CapEx (on-net)."
    )

    default = next(
        (i for i, o in enumerate(p.options)
         if o.name == p.best.name and o.quote.cloud_provider == preferred),
        p.options.index(p.best),
    )
    choice: int = st.selectbox(
        "Choose your plan", range(len(p.options)), index=default,
        format_func=lambda i: _option_label(p.options[i]),
        key=f"nc_choice_{p.options[0].quote.quote_id}",
    )
    return choice


# --- Account creation + provisioning ---------------------------------------------
def _onboard(p: Proposal, option: PlanOption, ctx: LumenContext) -> None:
    """Create the account (once), then provision via the existing Codex flow."""
    state = st.session_state
    state.nc_error = ""
    if not _budget_ok():
        return
    account = state.nc_account
    if account is None or account.get("status") != "SUCCESS":
        account = create_account(p.requirement, p.location)
        state.nc_account = account
        if account.get("status") != "SUCCESS":
            state.nc_error = ("Account creation rejected: "
                              f"{account.get('errors', account.get('message'))}")
            return
    try:
        result = generate_provisioning_payload(
            [ChatMessage(role="user", content=config.APPROVAL_TOKEN)],
            option.quote, ctx,
        )
    except AgentError as exc:
        state.spent_usd += exc.usage.cost_usd
        state.nc_error = f"Provisioning failed: {exc}"
        return
    except Exception:  # noqa: BLE001 - last-resort guard; full trace is logged
        logger.exception("Unexpected error provisioning a new customer")
        state.nc_error = "Unexpected error. See logs/broker.log for details."
        return
    state.spent_usd += result.usage.cost_usd
    state.nc_provisioning = result
    state.nc_ordered_quote = option.quote


def render_new_customer(
    ctx: LumenContext,
    render_quote: Callable[[Quote], None],
    render_provisioning: Callable[[ProvisioningResult], None],
) -> None:
    """Entry point called by ``app.main`` when "New customer" is selected."""
    state = st.session_state
    for key, value in _STATE_DEFAULTS.items():
        state.setdefault(key, value)

    intake_col, plan_col = st.columns([2, 3], gap="large")
    with intake_col:
        st.subheader("🆕 New customer — site survey")
        text = st.text_area("Describe your requirement", placeholder=EXAMPLE,
                            height=120)
        if st.button("✨ Auto-fill form from description",
                     disabled=not text.strip()):
            _autofill(text, ctx)
        requirement = _intake_form(ctx)
        if requirement is not None:
            try:
                with st.spinner("Checking Lumen serviceability and pricing…"):
                    state.nc_proposal = build_proposal(ctx, requirement)
            except QuoteError as exc:
                st.error(str(exc))
            else:
                state.update(nc_account=None, nc_provisioning=None,
                             nc_ordered_quote=None, nc_error="")

    with plan_col:
        proposal: Proposal | None = state.nc_proposal
        if proposal is None:
            st.info("Available services, site sizing and the best price plan "
                    "across AWS, GCP and Azure will appear here.")
            return
        choice = _render_proposal(proposal)
        if choice is None:
            return

        result: ProvisioningResult | None = state.nc_provisioning
        account = state.nc_account
        if result is not None and result.succeeded:
            render_quote(state.nc_ordered_quote)
            st.success(f"✅ Account `{account['account_id']}` created and "
                       "network provisioned.")
            render_provisioning(result)
            return

        option = proposal.options[choice]
        render_quote(option.quote)
        if account and account.get("status") == "SUCCESS":
            st.info(f"Account `{account['account_id']}` created.")
        label = ("🔁 Retry provisioning" if account
                 else "🚀 Create account & provision this plan")
        if st.button(label, type="primary"):
            with st.spinner("Creating the account and provisioning via "
                            f"{config.CODEX_MODEL}…"):
                _onboard(proposal, option, ctx)
            st.rerun()
        if state.nc_error:
            st.error(f"⚠️ {state.nc_error}")
        if result is not None:
            render_provisioning(result)
