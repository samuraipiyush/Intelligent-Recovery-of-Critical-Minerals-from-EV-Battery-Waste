import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from models.composition import compute_composition
from models.recovery import compute_recovery
from models.economics import compute_revenue, compute_profit
from models.process import select_process

st.set_page_config(
    page_title="EV Battery Recycling Platform",
    page_icon="♻️",
    layout="wide",
)

# ── session state defaults ────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {role, content}
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = []   # raw messages for API
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Input Parameters")
battery = st.sidebar.selectbox("Battery Type", ["NMC", "LFP", "LCO"])
mass = st.sidebar.number_input("Mass (kg)", min_value=1.0, value=1000.0, step=50.0)
cost = st.sidebar.number_input(
    "Processing Cost (₹)", min_value=0.0, value=500_000.0, step=10_000.0
)
run = st.sidebar.button("Run Analysis", use_container_width=True)

st.title("EV Battery Recycling — Intelligent Recovery Platform")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_core, tab_llm, tab_twin = st.tabs(
    ["Core Analysis", "AI Assistant", "Digital Twin"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — CORE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab_core:
    if run:
        comp = compute_composition(battery, mass)
        recovered = compute_recovery(comp)
        revenue, breakdown = compute_revenue(recovered)
        profit = compute_profit(revenue, cost)
        process, reason = select_process(battery, recovered)

        st.session_state.last_result = {
            "battery": battery,
            "mass": mass,
            "cost": cost,
            "comp": comp,
            "recovered": recovered,
            "revenue": revenue,
            "breakdown": breakdown,
            "profit": profit,
            "process": process,
            "reason": reason,
        }

    res = st.session_state.last_result

    if res is None:
        st.info("Set parameters in the sidebar and click **Run Analysis**.")
    else:
        # KPI metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Battery Type", res["battery"])
        col2.metric("Total Revenue", f"₹ {res['revenue']:,.0f}")
        col3.metric(
            "Profit",
            f"₹ {res['profit']:,.0f}",
            delta=f"{'▲' if res['profit'] >= 0 else '▼'} {abs(res['profit']):,.0f}",
            delta_color="normal",
        )

        st.subheader("Process Recommendation")
        st.success(f"**{res['process']}** — {res['reason']}")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Composition vs Recovered (kg)")
            metals = list(res["comp"].keys())
            fig = go.Figure()
            fig.add_bar(name="In Battery", x=metals, y=[res["comp"][m] for m in metals])
            fig.add_bar(name="Recovered", x=metals, y=[res["recovered"][m] for m in metals])
            fig.update_layout(barmode="group", height=350, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Revenue Breakdown (₹)")
            fig2 = px.pie(
                names=list(res["breakdown"].keys()),
                values=list(res["breakdown"].values()),
                hole=0.35,
                height=350,
            )
            fig2.update_layout(margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Detailed Numbers"):
            df = pd.DataFrame(
                {
                    "Metal": list(res["comp"].keys()),
                    "In Battery (kg)": [f"{v:.3f}" for v in res["comp"].values()],
                    "Recovered (kg)": [
                        f"{res['recovered'][m]:.3f}" for m in res["comp"]
                    ],
                    "Revenue (₹)": [
                        f"₹ {res['breakdown'].get(m, 0):,.2f}" for m in res["comp"]
                    ],
                }
            )
            st.dataframe(df, use_container_width=True)
            st.write(f"**Total Revenue:** ₹ {res['revenue']:,.2f}")
            st.write(f"**Processing Cost:** ₹ {res['cost']:,.2f}")
            st.write(f"**Profit:** ₹ {res['profit']:,.2f}")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI ASSISTANT (LLM)
# ════════════════════════════════════════════════════════════════════════════════
with tab_llm:
    st.subheader("AI-Powered Decision Support")
    st.caption(
        "Ask natural-language questions about recovery routes, economics, or "
        "what-if scenarios. The assistant calls the same backend models as the "
        "Core Analysis tab."
    )

    try:
        from models.llm_agent import chat as llm_chat, active_provider
        provider_label = {"groq": "Groq (Llama 3.3 · free)", "anthropic": "Anthropic Claude"}.get(
            active_provider(), active_provider()
        )
        st.info(f"LLM provider: **{provider_label}**", icon="🤖")
        llm_available = True
    except Exception as e:
        llm_available = False
        st.error(f"LLM module unavailable: {e}")

    if llm_available:
        # render history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about EV battery recycling…"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.llm_messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        reply, updated_msgs = llm_chat(st.session_state.llm_messages)
                        st.session_state.llm_messages = updated_msgs
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": reply}
                        )
                        st.markdown(reply)
                    except Exception as e:
                        err = f"Error calling LLM: {e}"
                        st.error(err)

        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.llm_messages = []
            st.rerun()

        with st.expander("Example questions"):
            st.markdown("""
- *What is the best recovery route for 500 kg NMC waste with ₹2,00,000 processing cost?*
- *How can I maximise cobalt recovery from LCO batteries?*
- *What happens to profit if lithium price drops by 20% for 1000 kg NMC?*
- *Compare hydrometallurgy vs pyrometallurgy for LFP batteries.*
            """)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — DIGITAL TWIN
# ════════════════════════════════════════════════════════════════════════════════
with tab_twin:
    st.subheader("Digital Twin — Dynamic Process Simulation")
    st.caption(
        "Simulates time-dependent mass recovery through Leaching → Separation → "
        "Metal Recovery units using a first-order Arrhenius ODE model."
    )

    try:
        from models.digital_twin import run_simulation, sensitivity_over_temperature
        dt_available = True
    except Exception as e:
        dt_available = False
        st.error(f"Digital Twin module unavailable: {e}")

    if dt_available:
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown("**Simulation Controls**")
            dt_battery = st.selectbox("Battery Type", ["NMC", "LFP", "LCO"], key="dt_bat")
            dt_mass = st.slider("Batch Mass (kg)", 100, 5000, 1000, 100, key="dt_mass")
            dt_feed = st.slider(
                "Feed Rate (kg/min)", 1.0, 50.0, 10.0, 1.0, key="dt_feed"
            )
            dt_temp = st.slider(
                "Reactor Temperature (°C)", 40, 90, 60, 5, key="dt_temp"
            )
            dt_eff = st.slider(
                "Efficiency Scale", 0.5, 1.5, 1.0, 0.05, key="dt_eff",
                help="Multiplier on base recovery efficiencies"
            )
            dt_time = st.slider(
                "Simulation Duration (min)", 30, 240, 120, 10, key="dt_time"
            )
            run_twin = st.button("Run Simulation", use_container_width=True, key="run_dt")

        with col_b:
            if run_twin:
                with st.spinner("Simulating…"):
                    sim = run_simulation(
                        dt_battery, dt_mass, dt_feed, dt_temp, dt_eff, dt_time
                    )

                time = sim["time"]
                metals = list(sim["recovered"].keys())

                # time-series recovery plot
                fig_ts = go.Figure()
                for metal in metals:
                    fig_ts.add_scatter(
                        x=time,
                        y=sim["recovered"][metal],
                        mode="lines",
                        name=metal,
                    )
                fig_ts.update_layout(
                    title="Cumulative Recovered Mass Over Time",
                    xaxis_title="Time (min)",
                    yaxis_title="Recovered Mass (kg)",
                    height=380,
                )
                st.plotly_chart(fig_ts, use_container_width=True)

                # final recovery bar
                fig_bar = px.bar(
                    x=list(sim["final"].keys()),
                    y=list(sim["final"].values()),
                    labels={"x": "Metal", "y": "Final Recovered (kg)"},
                    title="Final Recovered Quantities",
                    height=280,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("**Final Recovered Masses**")
                for m, v in sim["final"].items():
                    pct = v / sim["composition"][m] * 100
                    st.write(f"- **{m}**: {v:.3f} kg &nbsp; ({pct:.1f}% of input)")
            else:
                st.info("Set controls and click **Run Simulation**.")

        # Temperature sensitivity section
        st.markdown("---")
        st.subheader("Temperature Sensitivity")
        st.caption(
            "Final recovered mass per metal across the temperature range "
            "(fixed 120-min simulation, default efficiency scale)."
        )
        run_sens = st.button("Run Temperature Sensitivity", key="run_sens")

        if run_sens:
            sens_mass = st.session_state.get("dt_mass", 1000)
            sens_feed = st.session_state.get("dt_feed", 10.0)
            with st.spinner("Running sensitivity…"):
                sens = sensitivity_over_temperature(
                    dt_battery, sens_mass, sens_feed, (40, 90), 10
                )

            df_sens = pd.DataFrame(sens["final_per_temp"])
            df_sens.insert(0, "Temperature (°C)", sens["temperatures"])

            fig_sens = px.line(
                df_sens,
                x="Temperature (°C)",
                y=[c for c in df_sens.columns if c != "Temperature (°C)"],
                title="Final Recovery vs Reactor Temperature",
                labels={"value": "Recovered (kg)", "variable": "Metal"},
                height=380,
            )
            st.plotly_chart(fig_sens, use_container_width=True)
