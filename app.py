"""
app.py
──────
AI-Driven Game Theoretic Model for Adaptive Cybersecurity Defense

Main Streamlit application entry point.

Dashboard Layout:
  Sidebar  — all game configuration inputs
  Tab 1    — 🌐 Network Topology
  Tab 2    — 🎯 Game Analysis (Nash + Pareto)
  Tab 3    — 🔄 Simulation (Repeated game + plots)
  Tab 4    — 🤖 AI Insights (Q-learner + pattern recognition)
  Tab 5    — ⚖️ Comparison (Nash vs LP)
  Tab 6    — 💾 Export

Run with:
  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import time

# ─── Engine ──────────────────────────────────────────────────────────────────
from engine.game_theory import (
    create_game,
    compute_nash_equilibria,
    classify_equilibria,
    compute_pareto_optimal,
    get_payoff_at,
)
from engine.network_model import build_default_network, NetworkModel
from engine.simulation import run_simulation, run_best_response_dynamics
from engine.optimization import compare_nash_vs_lp

# ─── AI ──────────────────────────────────────────────────────────────────────
from ai.q_learning import QLearningAgent
from ai.pattern_recognition import AttackPatternClassifier, generate_synthetic_history

# ─── UI ──────────────────────────────────────────────────────────────────────
from ui.network_viz import render_network
from ui.charts import (
    plot_strategy_evolution,
    plot_payoff_evolution,
    plot_pareto_frontier,
    plot_comparison_bar,
    plot_q_table,
    plot_attack_proba,
    plot_reward_history,
    plot_payoff_matrix_heatmap,
)
from ui.matrix_editor import (
    SCENARIO_PRESETS,
    build_default_matrices,
    render_matrix_editor,
)
from utils.export import (
    simulation_to_csv,
    pareto_to_csv,
    equilibria_to_csv,
    config_to_csv,
    export_scenario_json,
    parse_scenario_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CyberGame — Game Theoretic Defense Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

/* Global Reset */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F8FAFC;
    background-color: #090A0B !important;
}

/* Global App Background Override */
.stApp {
    background: #090A0B !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', -apple-system, sans-serif;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #FFFFFF;
}

/* Header */
.app-header {
    background: #111318;
    border: 1px solid #272A30;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 32px;
}
.app-title {
    font-size: 24px;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: -0.01em;
    margin: 0;
}
.app-subtitle {
    font-size: 14px;
    color: #94A3B8;
    font-weight: 400;
    margin-top: 4px;
}

/* Metric Cards */
.metric-card {
    background: #111318;
    border: 1px solid #272A30;
    border-radius: 8px;
    padding: 20px;
    text-align: left;
    transition: background 0.15s ease;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card:hover { 
    background: #16181D;
}
.metric-val { 
    font-size: 32px; 
    font-weight: 600; 
    color: #FFFFFF; 
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
    line-height: 1.1;
}
.metric-lbl { 
    font-size: 12px; 
    color: #94A3B8; 
    font-weight: 500;
    text-transform: uppercase; 
    letter-spacing: 0.05em; 
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #090A0B;
    border-right: 1px solid #272A30;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-top: 16px;
    margin-bottom: 8px;
}

/* Inputs & Form Elements */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
    background-color: #090A0B !important;
    border: 1px solid #272A30 !important;
    color: #F8FAFC !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
    border-color: #635BFF !important;
    box-shadow: 0 0 0 2px rgba(99, 91, 255, 0.25) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    background-color: transparent !important;
    padding: 10px 16px !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"]:hover {
    color: #E2E8F0 !important;
}
button[aria-selected="true"] { 
    color: #FFFFFF !important; 
    border-bottom: 2px solid #FFFFFF !important; 
}

/* Standard Buttons */
.stButton > button {
    background: #111318 !important;
    border: 1px solid #272A30 !important;
    color: #F8FAFC !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #16181D !important;
    border-color: #334155 !important;
}
.stButton > button:active { 
    background: #090A0B !important;
}

/* Primary Button (Stripe Blurple) */
.stButton > button[kind="primary"] {
    background: #635BFF !important;
    border: 1px solid #635BFF !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 4px rgba(99, 91, 255, 0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #5448E5 !important;
    border-color: #5448E5 !important;
}

/* Expanders */
.streamlit-expanderHeader { 
    background: #111318 !important; 
    border: 1px solid #272A30 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
.streamlit-expanderHeader:hover {
    background: #16181D !important;
}
.streamlit-expanderContent {
    border: 1px solid #272A30;
    border-top: none;
    border-radius: 0 0 6px 6px;
}

/* DataFrames */
.stDataFrame {
    border-radius: 8px;
    border: 1px solid #272A30;
    overflow: hidden;
}

/* Utils & Chips */
hr { border-color: #272A30; margin: 24px 0; }
code, .stCode { font-family: 'Fira Code', monospace; background: #16181D !important; border: 1px solid #272A30 !important; border-radius: 4px !important; }

/* Status chips */
.chip {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.chip-pure   { background: rgba(16, 185, 129, 0.1); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.2); }
.chip-mixed  { background: rgba(99, 91, 255, 0.1); color: #635BFF; border: 1px solid rgba(99, 91, 255, 0.2); }
.chip-pareto { background: rgba(56, 189, 248, 0.1); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.2); }

.live-status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #10B981; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    """Initialise Streamlit session-state keys on first load."""
    if "network" not in st.session_state:
        st.session_state.network = build_default_network()

    if "sim_df" not in st.session_state:
        st.session_state.sim_df = None

    if "brd_df" not in st.session_state:
        st.session_state.brd_df = None

    if "equilibria" not in st.session_state:
        st.session_state.equilibria = []

    if "classified_eqs" not in st.session_state:
        st.session_state.classified_eqs = []

    if "pareto" not in st.session_state:
        st.session_state.pareto = []

    if "q_agent" not in st.session_state:
        st.session_state.q_agent = None

    if "atk_classifier" not in st.session_state:
        st.session_state.atk_classifier = None

    if "comparison" not in st.session_state:
        st.session_state.comparison = None

    if "last_A" not in st.session_state:
        st.session_state.last_A = None

    if "last_B" not in st.session_state:
        st.session_state.last_B = None

    if "human_score" not in st.session_state:
        st.session_state.human_score = 0.0

    if "ai_score" not in st.session_state:
        st.session_state.ai_score = 0.0

    if "human_history" not in st.session_state:
        st.session_state.human_history = []

    if "ai_history" not in st.session_state:
        st.session_state.ai_history = []

    if "play_logs" not in st.session_state:
        st.session_state.play_logs = []


init_session()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <p class="app-title">🛡️ AI-Driven Game Theoretic Cybersecurity Defense</p>
  <p class="app-subtitle">
    Interactive simulation platform · Attacker vs Defender · Nash Equilibria · Adaptive AI
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Game Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.markdown("---")

    # ── Scenario preset & Import ──────────────────────────────────────────
    st.markdown("### 📋 Scenario Template")
    
    uploaded_file = st.file_uploader("📥 Load Custom Scenario (JSON)", type=["json"])
    imported_preset = None
    if uploaded_file is not None:
        try:
            imported_preset = parse_scenario_json(uploaded_file.getvalue())
            st.success("✅ Loaded from JSON!")
        except Exception as e:
            st.error("❌ Failed to parse JSON")

    if imported_preset is not None:
        scenario_name = "Imported JSON"
        use_preset = True
        preset = imported_preset
    else:
        scenario_name = st.selectbox(
            "Choose a template",
            options=["Custom Builder"] + list(SCENARIO_PRESETS.keys()),
            key="scenario_select",
        )
        if scenario_name != "Custom Builder":
            preset = SCENARIO_PRESETS[scenario_name]
            st.info(preset.get("description", ""))
            use_preset = True
        else:
            use_preset = False

    st.markdown("---")

    # ── Strategy counts ────────────────────────────────────────────────────
    st.markdown("### 🎲 Strategy Matrix")

    if use_preset:
        n_def = len(preset["def_labels"])
        n_atk = len(preset["atk_labels"])
        def_labels = preset["def_labels"]
        atk_labels = preset["atk_labels"]
        st.write(f"**Defender:** {n_def} strategies")
        st.write(f"**Attacker:** {n_atk} strategies")
    else:
        n_def = st.slider("Defender strategies", 2, 6, 2, key="n_def")
        n_atk = st.slider("Attacker strategies", 2, 6, 2, key="n_atk")

        st.markdown("**Defender Labels:**")
        def_labels = []
        for i in range(n_def):
            lbl = st.text_input(
                f"D{i+1}", value=f"Defend_{i+1}",
                key=f"def_lbl_{i}", label_visibility="collapsed"
            )
            def_labels.append(lbl or f"D{i+1}")

        st.markdown("**Attacker Labels:**")
        atk_labels = []
        for j in range(n_atk):
            lbl = st.text_input(
                f"A{j+1}", value=f"Attack_{j+1}",
                key=f"atk_lbl_{j}", label_visibility="collapsed"
            )
            atk_labels.append(lbl or f"A{j+1}")

    st.markdown("---")
    # ── Payoff matrices ────────────────────────────────────────────────────
    st.markdown("### 💰 Payoff Values")

    if use_preset:
        A = preset["A"].copy()
        B = preset["B"].copy()
    else:
        prev_A = st.session_state.last_A
        prev_B = st.session_state.last_B
        if (prev_A is not None and prev_A.shape == (n_def, n_atk)):
            A_init = prev_A
            B_init = prev_B
        else:
            A_init, B_init = build_default_matrices(n_def, n_atk)

        try:
            A = render_matrix_editor(
                "Defender Payoffs",
                A_init, def_labels, atk_labels, key="matrix_A"
            )
            B = render_matrix_editor(
                "Attacker Payoffs",
                B_init, def_labels, atk_labels, key="matrix_B"
            )
            # Security: Sanitize user input (prevents NaN or Infinity crashes)
            A = np.nan_to_num(A.astype(float), nan=0.0, posinf=9999.0, neginf=-9999.0)
            B = np.nan_to_num(B.astype(float), nan=0.0, posinf=9999.0, neginf=-9999.0)
        except Exception:
            st.error("Input overflow detected. Reverting grid.")
            A, B = A_init, B_init

    st.session_state.last_A = A
    st.session_state.last_B = B

    st.markdown("---")

    # ── Game parameters ────────────────────────────────────────────────────
    st.markdown("### 🔬 Advanced Settings")
    n_rounds  = st.slider("Simulation rounds", 20, 1000, 100, step=10)
    use_ai    = st.toggle("Enable Q-Learning AI", value=True)
    use_brd   = st.toggle("Best-Response Engine", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("▶ Run Full Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    json_str = export_scenario_json(A, B, def_labels, atk_labels)
    st.download_button("📥 Export JSON", data=json_str, file_name="cybergame_scenario.json", mime="application/json", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation (runs when button pressed or matrices change)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_nash_cached(A: np.ndarray, B: np.ndarray):
    game = create_game(A, B)
    eqs = compute_nash_equilibria(game)
    cls = classify_equilibria(eqs)
    par = compute_pareto_optimal(A, B)
    return eqs, cls, par

@st.cache_data(show_spinner=False)
def run_brd_cached(A: np.ndarray, B: np.ndarray, rounds: int):
    return run_best_response_dynamics(A, B, rounds=rounds)

@st.cache_data(show_spinner=False)
def solve_lp_cached(A: np.ndarray, B: np.ndarray, nash_def_payoff, nash_atk_payoff):
    return compare_nash_vs_lp(A, B, nash_def_payoff, nash_atk_payoff)

def run_analysis(A, B, n_rounds, use_ai, use_brd, def_labels, atk_labels):
    """Run the full game-theoretic and simulation analysis pipeline."""
    n_def, n_atk = A.shape

    with st.spinner("⚙️ Computing Nash Equilibria…"):
        equilibria, classified, pareto = compute_nash_cached(A, B)

    st.session_state.equilibria     = equilibria
    st.session_state.classified_eqs = classified
    st.session_state.pareto         = pareto

    # ── Build/reset AI agents ─────────────────────────────────────────────
    q_agent = None
    if use_ai:
        q_agent = QLearningAgent(n_states=n_atk, n_actions=n_def, epsilon=1.0)
        st.session_state.q_agent = q_agent

    # ── Run main simulation ───────────────────────────────────────────────
    with st.spinner("🔄 Running simulation…"):
        sim_df = run_simulation(
            A, B, equilibria, rounds=n_rounds,
            ai_agent=q_agent, use_best_response=False,
        )
    st.session_state.sim_df = sim_df

    # ── Best-response dynamics ─────────────────────────────────────────────
    if use_brd:
        with st.spinner("📈 Running best-response dynamics…"):
            brd_df = run_brd_cached(A, B, n_rounds)
        st.session_state.brd_df = brd_df

    # ── Attack pattern classifier ─────────────────────────────────────────
    with st.spinner("🤖 Training attack pattern classifier…"):
        atk_history_synth = generate_synthetic_history(n_atk, 300)
        # Also include actual simulation history
        if sim_df is not None:
            atk_history_synth += list(sim_df["attacker_strategy"])
        classifier = AttackPatternClassifier(n_strategies=n_atk, window_size=3)
        classifier.fit(atk_history_synth)
        st.session_state.atk_classifier = classifier

    # ── LP comparison ─────────────────────────────────────────────────────
    nash_def_payoff = None
    nash_atk_payoff = None
    if equilibria:
        game = create_game(A, B)
        nash_def_payoff, nash_atk_payoff = get_payoff_at(game, *equilibria[0])

    with st.spinner("⚖️ Solving LP comparison…"):
        comparison = solve_lp_cached(A, B, nash_def_payoff, nash_atk_payoff)
    st.session_state.comparison = comparison

    return classified, pareto, sim_df


# Auto-run if button clicked
if run_clicked:
    classified, pareto, sim_df = run_analysis(
        A, B, n_rounds, use_ai, use_brd, def_labels, atk_labels
    )
    st.success("✅ Analysis complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Retrieve state
# ─────────────────────────────────────────────────────────────────────────────

equilibria     = st.session_state.equilibria
classified_eqs = st.session_state.classified_eqs
pareto         = st.session_state.pareto
sim_df         = st.session_state.sim_df
brd_df         = st.session_state.brd_df
q_agent        = st.session_state.q_agent
atk_classifier = st.session_state.atk_classifier
comparison     = st.session_state.comparison
network        = st.session_state.network
n_def, n_atk   = A.shape


# ─────────────────────────────────────────────────────────────────────────────
# Top-level KPI strip
# ─────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(equilibria)}</div>
        <div class="metric-lbl">Nash Equilibria Found</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(pareto)}</div>
        <div class="metric-lbl">Pareto-Optimal Outcomes</div>
    </div>""", unsafe_allow_html=True)

with k3:
    rounds_done = len(sim_df) if sim_df is not None else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{rounds_done}</div>
        <div class="metric-lbl">Simulation Rounds</div>
    </div>""", unsafe_allow_html=True)

with k4:
    net_summary = network.state_summary()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{network.total_value()}</div>
        <div class="metric-lbl">Total Network Asset Value</div>
    </div>""", unsafe_allow_html=True)

with k5:
    ai_status = "Active" if (q_agent is not None) else "Off"
    ai_color  = "#39FF14" if ai_status == "Active" else "#8892a4"
    anim_cls  = "live-status" if ai_status == "Active" else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val {anim_cls}" style="color:{ai_color}">{ai_status}</div>
        <div class="metric-lbl">AI Defender (Q-Learning)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

TAB_ICONS = ["Network", "Strategic Analysis", "Simulation Engine", "AI Observer", "Comparison", "Export", "Arena", "Stream Telemetry"]
tabs = st.tabs(TAB_ICONS)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — NETWORK TOPOLOGY
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("### 🌐 Network Topology Visualization")
    st.caption("Interactive drag-and-drop network. Colors reflect device state.")

    col_net, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown("**Node Controls**")
        node_ids   = network.node_ids()
        node_names = {nid: network.G.nodes[nid]["label"] for nid in node_ids}

        if node_ids:
            selected_node = st.selectbox(
                "Select node", node_ids,
                format_func=lambda x: node_names[x],
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔴 Attack", use_container_width=True):
                    network.attack_node(selected_node)
                    st.rerun()
            with c2:
                if st.button("🟢 Defend", use_container_width=True):
                    network.defend_node(selected_node)
                    st.rerun()

            if st.button("⚪ Reset Node", use_container_width=True):
                network.reset_node(selected_node)
                st.rerun()

            if st.button("🔄 Reset All Nodes", use_container_width=True):
                network.reset_all()
                st.rerun()

        st.markdown("---")
        st.markdown("**Network Legend**")
        legend_items = [
            ("🔵", "Normal",      "#4A90D9"),
            ("🔴", "Attacked",    "#E74C3C"),
            ("🟢", "Defended",    "#2ECC71"),
            ("🟡", "Compromised", "#F39C12"),
        ]
        for icon, label, color in legend_items:
            st.markdown(
                f'<span style="color:{color}">■</span> {label}',
                unsafe_allow_html=True
            )

        st.markdown("---")
        summary = network.state_summary()
        st.markdown("**State Summary**")
        for state, count in summary.items():
            st.write(f"• {state.capitalize()}: **{count}**")

        comp_val = network.compromised_value()
        total_val = network.total_value()
        if total_val > 0:
            pct = comp_val / total_val * 100
            st.metric("Asset Risk Exposure", f"{pct:.1f}%",
                       delta=f"-{comp_val} value at risk",
                       delta_color="inverse")

    with col_net:
        render_network(network, height=520)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — GAME ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("### 🎯 Game Analysis — Nash Equilibria & Pareto Optimality")

    if not classified_eqs:
        st.info("▶ Press **Run Full Analysis** in the sidebar to compute equilibria.")
    else:
        # ── Nash Equilibria ────────────────────────────────────────────────
        st.markdown("#### Nash Equilibria")
        for idx, eq in enumerate(classified_eqs):
            with st.expander(
                f"Equilibrium {idx+1}  —  "
                f"{'🟢 Pure Strategy' if eq['type']=='Pure' else '🔵 Mixed Strategy'}", 
                expanded=(idx == 0)
            ):
                col_d, col_a = st.columns(2)
                with col_d:
                    st.markdown("**Defender Mixed Strategy (σ_D)**")
                    for i, (lbl, p) in enumerate(zip(def_labels, eq["sigma_defender"])):
                        st.progress(float(p), text=f"{lbl}: {p:.4f}")
                with col_a:
                    st.markdown("**Attacker Mixed Strategy (σ_A)**")
                    for j, (lbl, p) in enumerate(zip(atk_labels, eq["sigma_attacker"])):
                        st.progress(float(p), text=f"{lbl}: {p:.4f}")

                # Compute payoffs at this equilibrium
                try:
                    game = create_game(A, B)
                    d_pay, a_pay = get_payoff_at(game, eq["sigma_defender"], eq["sigma_attacker"])
                    pc1, pc2 = st.columns(2)
                    pc1.metric("Defender Expected Payoff", f"{d_pay:.4f}")
                    pc2.metric("Attacker Expected Payoff",  f"{a_pay:.4f}")
                except Exception:
                    pass

        st.markdown("---")

        # ── Payoff matrix heatmaps ─────────────────────────────────────────
        st.markdown("#### Payoff Matrix Heatmap")
        st.plotly_chart(
            plot_payoff_matrix_heatmap(A, B, def_labels, atk_labels),
            use_container_width=True,
        )

        st.markdown("---")

        # ── Pareto optimality ──────────────────────────────────────────────
        st.markdown("#### Pareto-Optimal Outcomes")
        if pareto:
            st.plotly_chart(
                plot_pareto_frontier(pareto, A, B),
                use_container_width=True,
            )
            pareto_df = pd.DataFrame(pareto)
            pareto_df.columns = ["Defender Strategy", "Attacker Strategy",
                                  "Defender Payoff",   "Attacker Payoff"]
            pareto_df["Defender Strategy"] = pareto_df["Defender Strategy"].apply(
                lambda i: def_labels[int(i)] if int(i) < len(def_labels) else f"S{i+1}"
            )
            pareto_df["Attacker Strategy"] = pareto_df["Attacker Strategy"].apply(
                lambda j: atk_labels[int(j)] if int(j) < len(atk_labels) else f"S{j+1}"
            )
            st.dataframe(pareto_df, use_container_width=True)
        else:
            st.warning("No Pareto-optimal pure-strategy outcomes identified.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIMULATION
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### 🔄 Repeated Game Simulation")

    if sim_df is None:
        st.info("▶ Press **Run Full Analysis** in the sidebar to start the simulation.")
    else:
        # ── Key metrics ────────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Avg Defender Payoff", f"{sim_df['defender_payoff'].mean():.3f}")
        mc2.metric("Avg Attacker Payoff", f"{sim_df['attacker_payoff'].mean():.3f}")
        mc3.metric("Defender Wins",
                   f"{(sim_df['defender_payoff'] > sim_df['attacker_payoff']).sum()}")
        mc4.metric("Attacker Wins",
                   f"{(sim_df['attacker_payoff'] > sim_df['defender_payoff']).sum()}")

        st.markdown("---")

        # ── Strategy evolution ─────────────────────────────────────────────
        st.subheader("Strategy Probability Evolution")
        st.plotly_chart(
            plot_strategy_evolution(sim_df, n_def, n_atk),
            use_container_width=True,
            key="sim_strategy_evolution"
        )

        # ── Payoff convergence ─────────────────────────────────────────────
        st.subheader("Payoff Convergence")
        st.plotly_chart(
            plot_payoff_evolution(sim_df),
            use_container_width=True,
            key="sim_payoff_convergence"
        )

        # ── Best-response dynamics ─────────────────────────────────────────
        if brd_df is not None:
            st.markdown("---")
            st.subheader("Best-Response Dynamics")
            st.caption("Each player best-responds to the opponent's empirical frequency. "
                        "Convergence indicates proximity to Nash Equilibrium.")
            st.plotly_chart(
                plot_payoff_evolution(brd_df),
                use_container_width=True,
                key="brd_payoff_convergence"
            )
            st.plotly_chart(
                plot_strategy_evolution(brd_df, n_def, n_atk),
                use_container_width=True,
                key="brd_strategy_evolution"
            )

        # ── Raw data table ─────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋 View Raw Simulation Data"):
            st.dataframe(sim_df, use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("### 🤖 AI Adaptive Defense — Q-Learning & Pattern Recognition")

    ai_col1, ai_col2 = st.columns(2)

    # ── Q-Learning section ─────────────────────────────────────────────────
    with ai_col1:
        st.markdown("#### Q-Learning Defender Agent")

        if q_agent is None:
            st.info("Enable **Q-Learning Defender** in the sidebar and run analysis.")
        else:
            # Q-table heatmap
            q_table = q_agent.get_q_table()
            st.plotly_chart(
                plot_q_table(q_table, def_labels, atk_labels),
                use_container_width=True,
            )

            # Policy readout
            policy = q_agent.get_policy()
            st.markdown("**Learned Greedy Policy**")
            policy_df = pd.DataFrame({
                "When Attacker plays":  [atk_labels[s] for s in range(n_atk)],
                "Defender should play": [def_labels[int(a)] for a in policy],
            })
            st.dataframe(policy_df, use_container_width=True, hide_index=True)

            # Reward history
            if q_agent.reward_history:
                st.plotly_chart(
                    plot_reward_history(q_agent.reward_history),
                    use_container_width=True,
                )

            # Epsilon decay
            if q_agent.epsilon_history:
                eps_fig = plot_reward_history(q_agent.epsilon_history)
                eps_fig.update_layout(
                    title="Exploration Rate (ε) Decay",
                    yaxis_title="Epsilon",
                )
                st.plotly_chart(eps_fig, use_container_width=True)

    # ── Pattern Recognition section ────────────────────────────────────────
    with ai_col2:
        st.markdown("#### Attack Pattern Recognition")

        if atk_classifier is None:
            st.info("Run the analysis to train the attack pattern classifier.")
        else:
            trained_lbl = "✅ Trained" if atk_classifier.is_trained else "❌ Not Trained"
            st.markdown(f"**Classifier status:** {trained_lbl}")

            # Predict next attacker strategy
            st.markdown("---")
            st.markdown("**Predict Next Attack**")
            st.caption("Enter the last 3 attacker strategy indices (0-indexed):")

            window_input = []
            w_cols = st.columns(3)
            for wi, wc in enumerate(w_cols):
                val = wc.slider(f"t-{3-wi}", 0, n_atk - 1, 0, key=f"window_{wi}")
                window_input.append(val)

            if st.button("🔮 Predict Next Attacker Move", use_container_width=True):
                proba = atk_classifier.predict_proba(window_input)
                predicted = atk_classifier.predict(window_input)
                st.success(f"**Predicted:** {atk_labels[predicted]}")
                st.plotly_chart(
                    plot_attack_proba(proba, atk_labels),
                    use_container_width=True,
                )

            # Feature importances
            st.markdown("---")
            st.markdown("**Feature Importance (past moves)**")
            importances = atk_classifier.get_feature_importance()
            fi_df = pd.DataFrame({
                "Time Step": [f"t-{3-i}" for i in range(len(importances))],
                "Importance": importances,
            })
            st.bar_chart(fi_df.set_index("Time Step"))

            # Live prediction from simulation history
            if sim_df is not None and len(sim_df) >= 3:
                st.markdown("---")
                st.markdown("**Live Prediction from Simulation History**")
                last_window = list(sim_df["attacker_strategy"].tail(3))
                live_proba = atk_classifier.predict_proba(last_window)
                live_pred  = atk_classifier.predict(last_window)
                st.info(f"Based on last 3 rounds → Predicted next: **{atk_labels[live_pred]}**")
                st.plotly_chart(
                    plot_attack_proba(live_proba, atk_labels),
                    use_container_width=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARISON: NASH vs LP
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### ⚖️ Game Equilibrium vs. Centralized LP Optimization")
    st.caption(
        "Compares the **decentralized** Nash Equilibrium solution with the "
        "**centralized** Linear Programming optimal solution."
    )

    if comparison is None:
        st.info("▶ Run the full analysis to see the comparison.")
    else:
        # ── Comparison bar chart ───────────────────────────────────────────
        nash_def = comparison.get("nash_defender_payoff", 0) or 0
        nash_atk = comparison.get("nash_attacker_payoff", 0) or 0
        lp_def   = comparison.get("lp_defender_value",   0) or 0
        lp_atk   = comparison.get("lp_attacker_value",   0) or 0

        st.plotly_chart(
            plot_comparison_bar(nash_def, nash_atk, lp_def, lp_atk),
            use_container_width=True,
        )

        # ── Summary table ──────────────────────────────────────────────────
        st.markdown("#### Efficiency Comparison Table")
        comp_rows = []
        for player, ne_val, lp_val in [
            ("Defender", nash_def, lp_def),
            ("Attacker", nash_atk, lp_atk),
        ]:
            diff = lp_val - ne_val if (lp_val and ne_val) else None
            comp_rows.append({
                "Player":            player,
                "Nash Payoff":       f"{ne_val:.4f}",
                "LP Optimal Payoff": f"{lp_val:.4f}",
                "Efficiency Gap":    f"{diff:.4f}" if diff is not None else "N/A",
                "Interpretation":    (
                    "LP outperforms Nash" if (diff and diff > 0.001)
                    else "Equivalent" if diff is not None
                    else "N/A"
                ),
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # ── Optimal strategies ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### LP-Optimal Strategies")
        lp_col1, lp_col2 = st.columns(2)

        with lp_col1:
            st.markdown("**LP Optimal Defender Strategy**")
            if comparison.get("lp_defender_success"):
                lp_def_strat = comparison["lp_defender_strategy"]
                for i, (lbl, p) in enumerate(zip(def_labels, lp_def_strat)):
                    st.progress(float(np.clip(p, 0, 1)), text=f"{lbl}: {p:.4f}")
            else:
                st.error("LP solver did not converge.")

        with lp_col2:
            st.markdown("**LP Optimal Attacker Strategy**")
            if comparison.get("lp_attacker_success"):
                lp_atk_strat = comparison["lp_attacker_strategy"]
                for j, (lbl, p) in enumerate(zip(atk_labels, lp_atk_strat)):
                    st.progress(float(np.clip(p, 0, 1)), text=f"{lbl}: {p:.4f}")
            else:
                st.error("LP solver did not converge.")

        # ── Theory note ────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("ℹ️ About Nash vs LP"):
            st.markdown("""
            **Nash Equilibrium** is the outcome when both players act *independently*
            and rationally. Neither has incentive to deviate unilaterally — it is a stable
            equilibrium in a decentralized setting.

            **Linear Programming (LP)** finds the *minimax-optimal* strategy for each player
            in isolation, assuming a zero-sum structure. This represents the *centralised*
            optimum — the best a player can guarantee regardless of the opponent's action.

            For **zero-sum games**, both methods coincide (minimax theorem). For
            **general-sum games**, the LP optimum may differ slightly from Nash because the LP
            ignores the opponent's incentives. The efficiency gap measures this difference.
            """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPORT
# ════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("### 💾 Export Results")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("#### Game Configuration")
        config_csv = config_to_csv(A, B, def_labels, atk_labels)
        st.download_button(
            label="⬇ Download Payoff Matrices (CSV)",
            data=config_csv,
            file_name="cybergame_payoff_matrices.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if classified_eqs:
            eq_csv = equilibria_to_csv(classified_eqs)
            st.download_button(
                label="⬇ Download Nash Equilibria (CSV)",
                data=eq_csv,
                file_name="cybergame_nash_equilibria.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if pareto:
            pareto_csv = pareto_to_csv(pareto)
            st.download_button(
                label="⬇ Download Pareto Outcomes (CSV)",
                data=pareto_csv,
                file_name="cybergame_pareto_optimal.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with exp_col2:
        st.markdown("#### Simulation Results")
        if sim_df is not None:
            sim_csv = simulation_to_csv(sim_df)
            st.download_button(
                label="⬇ Download Simulation Data (CSV)",
                data=sim_csv,
                file_name="cybergame_simulation.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if brd_df is not None:
            brd_csv = simulation_to_csv(brd_df)
            st.download_button(
                label="⬇ Download Best-Response Dynamics (CSV)",
                data=brd_csv,
                file_name="cybergame_brd.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("#### Scenario Summary")
    summary_data = {
        "Scenario": scenario_name,
        "Defender Strategies": n_def,
        "Attacker Strategies": n_atk,
        "Nash Equilibria Found": len(equilibria),
        "Pareto Optimal Outcomes": len(pareto),
        "Simulation Rounds": len(sim_df) if sim_df is not None else 0,
        "AI Defender": "Q-Learning (Active)" if q_agent else "Disabled",
    }
    st.json(summary_data)


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — PLAY VS AI
# ════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown("### 🎮 Arena: Play vs AI")
    st.caption("You are the Attacker. The Q-Learning AI is the Defender. Try to maximize your payoff by outsmarting the model!")

    if q_agent is None:
        st.info("▶ Run the full analysis in the sidebar first to initialize the AI.")
    else:
        col_arena, col_score = st.columns([2, 1])

        with col_score:
            st.markdown("#### Scoreboard")
            s1, s2 = st.columns(2)
            s1.metric("Your Score (Attacker)", f"{st.session_state.human_score:.2f}")
            s2.metric("AI Score (Defender)", f"{st.session_state.ai_score:.2f}")

            if st.button("🔄 Reset Match", use_container_width=True):
                st.session_state.human_score = 0.0
                st.session_state.ai_score = 0.0
                st.session_state.human_history = []
                st.session_state.ai_history = []
                st.session_state.play_logs = []
                st.rerun()

        with col_arena:
            st.markdown("#### Choose Your Attack Strategy")
            attack_cols = st.columns(n_atk)
            for j in range(n_atk):
                with attack_cols[j]:
                    if st.button(f"⚔️ {atk_labels[j]}", key=f"atk_btn_{j}", use_container_width=True):
                        # 1. State is the human's last move
                        state = st.session_state.human_history[-1] if len(st.session_state.human_history) > 0 else 0
                        # 2. AI picks an action (greedy, but adapts quickly)
                        ai_action = q_agent.select_action(state)
                        # 3. Compute payoffs
                        def_payoff = float(A[ai_action, j])
                        atk_payoff = float(B[ai_action, j])
                        # 4. Update Q-learning formulation with the new info (state transition)
                        q_agent.update(state, ai_action, def_payoff, j)
                        
                        # 5. Save history
                        st.session_state.human_score += atk_payoff
                        st.session_state.ai_score += def_payoff
                        st.session_state.human_history.append(j)
                        st.session_state.ai_history.append(ai_action)
                        st.session_state.play_logs.insert(0, {
                            "Round": len(st.session_state.human_history),
                            "You Played": atk_labels[j],
                            "AI Played": def_labels[ai_action],
                            "Your Payoff": atk_payoff,
                            "AI Payoff": def_payoff
                        })
                        st.rerun()

        st.markdown("---")
        st.markdown("#### Match History")
        if st.session_state.play_logs:
            st.dataframe(pd.DataFrame(st.session_state.play_logs), use_container_width=True)
            
        # Feature showing what the AI expects next
        if atk_classifier and atk_classifier.is_trained:
            if len(st.session_state.human_history) >= 3:
                st.markdown("---")
                st.markdown("#### 🧠 Inside the AI's Brain")
                st.caption("The Pattern Classifier is actively trying to predict your next move based on your history.")
                last_window = st.session_state.human_history[-3:]
                live_proba = atk_classifier.predict_proba(last_window)
                live_pred  = atk_classifier.predict(last_window)
                st.info(f"The AI thinks you will play **{atk_labels[live_pred]}** next!")
                st.plotly_chart(
                    plot_attack_proba(live_proba, atk_labels),
                    use_container_width=True,
                    key="arena_predict"
                )
            else:
                st.markdown("---")
                st.info("🧠 Inside the AI's Brain: Make at least 3 moves for the AI to start predicting your strategy!")


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — LIVE TELEMETRY
# ════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    st.markdown("### 📡 Live Wargame Telemetry")
    st.caption("Execute a simulation in real-time. Watch the cumulative payoffs stack and observe the rapid back-and-forth action.")

    if q_agent is None:
        st.info("▶ Run the full analysis in the sidebar first to initialize the AI.")
    else:
        if "live_wargame_running" not in st.session_state:
            st.session_state.live_wargame_running = False

        if not st.session_state.live_wargame_running:
            if st.button("🚀 INITIATE LIVE WARGAME", use_container_width=True, type="primary"):
                st.session_state.live_wargame_running = True
                st.rerun()
        else:
            if st.button("🛑 ABORT SIMULATION", use_container_width=True):
                st.session_state.live_wargame_running = False
                st.rerun()

            st.markdown("---")
            # Setup Placeholders
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Live Cumulative Payoffs")
                chart_placeholder = st.empty()
            with c2:
                st.markdown("#### Event Feed")
                log_placeholder = st.empty()

            progress_bar = st.progress(0)

            # Initialize states for live execution
            live_rounds = 50
            def_score = 0
            atk_score = 0
            live_logs = []
            live_scores = []
            
            temp_history = [] 

            for r in range(1, live_rounds + 1):
                if not st.session_state.live_wargame_running:
                    break

                j = np.random.randint(0, n_atk)
                
                state = temp_history[-1] if len(temp_history) > 0 else 0
                ai_action = q_agent.select_action(state)
                
                def_payoff = float(A[ai_action, j])
                atk_payoff = float(B[ai_action, j])
                
                def_score += def_payoff
                atk_score += atk_payoff
                
                temp_history.append(j)
                
                live_scores.append({
                    "Round": r,
                    "Defender Score": def_score,
                    "Attacker Score": atk_score
                })
                
                live_logs.append(f"**R{r}:** Attacker deployed <span style='color:#FF003C;'>{atk_labels[j]}</span>...<br>AI blocked with <span style='color:#39FF14;'>{def_labels[ai_action]}</span>.")

                chart_placeholder.line_chart(pd.DataFrame(live_scores).set_index("Round"))
                
                display_logs = "<br><br>".join(live_logs[-6:])
                log_placeholder.markdown(f"<div style='font-family: JetBrains Mono; font-size: 0.85em; padding:10px; border:1px solid #00F3FF; background:rgba(0,0,0,0.5);'>{display_logs}</div>", unsafe_allow_html=True)
                
                progress_bar.progress(r / live_rounds)
                time.sleep(0.15)
                
            st.session_state.live_wargame_running = False
            if r == live_rounds:
                st.success("Wargame Complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:0.78rem;'>"
    "🛡️ CyberGame · AI-Driven Game Theoretic Cybersecurity Defense Simulator · "
    "Built with Streamlit + nashpy + NetworkX + scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
