"""
ui/charts.py
─────────────
All Plotly chart builders for the dashboard.

Provides:
  • plot_strategy_evolution(df, player)       → Plotly figure
  • plot_payoff_evolution(df)                 → Plotly figure
  • plot_pareto_frontier(pareto, A, B)        → Plotly figure
  • plot_comparison_bar(nash_vals, lp_vals)   → Plotly figure
  • plot_q_table(q_table, def_labels, atk_labels) → Plotly heatmap
  • plot_attack_proba(proba, strategy_names)  → Plotly bar chart
  • plot_reward_history(reward_history)       → Plotly line chart
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Color palette ────────────────────────────────────────────────────────────
DEFENDER_COLOR = "#10B981"   # muted emerald
ATTACKER_COLOR = "#EF4444"   # muted rose
ACCENT_COLOR   = "#635BFF"   # stripe blurple
PARETO_COLOR   = "#38BDF8"   # sky blue
SURFACE_BG     = "rgba(0,0,0,0)"
CARD_BG        = "rgba(0,0,0,0)"
GRID_COLOR     = "rgba(255,255,255,0.03)"
TEXT_COLOR     = "#94A3B8"
TITLE_COLOR    = "#FFFFFF"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family="'Inter', -apple-system, sans-serif"),
    title=dict(font=dict(family="'Inter', -apple-system, sans-serif", size=16, color=TITLE_COLOR)),
    legend=dict(bgcolor="rgba(17,19,24,0.8)", bordercolor="rgba(39,42,48,1)", borderwidth=1),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, gridwidth=1),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, gridwidth=1),
    margin=dict(l=40, r=20, t=60, b=40),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#111318",
        font_size=12,
        font_family="'Inter', sans-serif",
        bordercolor="#272A30"
    )
)

# ─────────────────────────────────────────────────────────────────────────────

def plot_strategy_evolution(df: pd.DataFrame, n_def: int, n_atk: int) -> go.Figure:
    """
    Line chart: mixed-strategy probability evolution over simulation rounds.

    Args:
        df:    Simulation DataFrame.
        n_def: Number of defender strategies.
        n_atk: Number of attacker strategies.

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Defender Strategy Probabilities", "Attacker Strategy Probabilities"),
        vertical_spacing=0.12,
    )

    # Defender prob columns
    def_cols = [f"def_prob_{i}" for i in range(n_def) if f"def_prob_{i}" in df.columns]
    for i, col in enumerate(def_cols):
        fig.add_trace(go.Scatter(
            x=df["round"], y=df[col],
            name=f"Defender S{i+1}",
            line=dict(width=2, shape='spline'),
            mode="lines",
        ), row=1, col=1)

    # Attacker prob columns
    atk_cols = [f"atk_prob_{j}" for j in range(n_atk) if f"atk_prob_{j}" in df.columns]
    for j, col in enumerate(atk_cols):
        fig.add_trace(go.Scatter(
            x=df["round"], y=df[col],
            name=f"Attacker S{j+1}",
            line=dict(width=2, dash="dot", shape='spline'),
            mode="lines",
        ), row=2, col=1)

    fig.update_layout(
        title="Strategy Distribution Over Time",
        height=420,
        **_PLOTLY_LAYOUT,
    )
    fig.update_yaxes(range=[0, 1], gridcolor=GRID_COLOR)
    return fig


def plot_payoff_evolution(df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis line chart of cumulative average payoffs over rounds.

    Args:
        df: Simulation DataFrame with cum_avg_defender, cum_avg_attacker.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["round"], y=df["cum_avg_defender"],
        name="Defender Avg Payoff",
        line=dict(color=DEFENDER_COLOR, width=2.5, shape='spline'),
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.1)",
    ))

    fig.add_trace(go.Scatter(
        x=df["round"], y=df["cum_avg_attacker"],
        name="Attacker Avg Payoff",
        line=dict(color=ATTACKER_COLOR, width=2.5, shape='spline'),
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.1)",
    ))

    fig.update_layout(
        title="Cumulative Average Payoffs (Convergence)",
        xaxis_title="Round",
        yaxis_title="Expected Payoff",
        height=350,
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_pareto_frontier(pareto: list, A: np.ndarray, B: np.ndarray) -> go.Figure:
    """
    Scatter plot of all pure-strategy outcomes with Pareto-optimal ones highlighted.

    Args:
        pareto: List of dicts from compute_pareto_optimal().
        A:      Defender payoff matrix.
        B:      Attacker payoff matrix.

    Returns:
        Plotly Figure.
    """
    m, n = A.shape
    all_def, all_atk, labels = [], [], []
    for i in range(m):
        for j in range(n):
            all_def.append(float(A[i, j]))
            all_atk.append(float(B[i, j]))
            labels.append(f"({i+1},{j+1})")

    pareto_def = [p["defender_payoff"] for p in pareto]
    pareto_atk = [p["attacker_payoff"] for p in pareto]
    pareto_labels = [f"({p['row']+1},{p['col']+1})" for p in pareto]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_def, y=all_atk,
        mode="markers+text",
        name="All Outcomes",
        marker=dict(size=10, color=GRID_COLOR, line=dict(color="#4A90D9", width=1.5)),
        text=labels,
        textposition="top center",
        textfont=dict(size=9, color=TEXT_COLOR),
    ))

    fig.add_trace(go.Scatter(
        x=pareto_def, y=pareto_atk,
        mode="markers+text",
        name="Pareto Optimal",
        marker=dict(size=16, color=PARETO_COLOR, symbol="star",
                    line=dict(color="#ffffff", width=1)),
        text=pareto_labels,
        textposition="top center",
        textfont=dict(size=10, color=PARETO_COLOR),
    ))

    fig.update_layout(
        title="Pareto Frontier — (Defender Payoff, Attacker Payoff)",
        xaxis_title="Defender Payoff",
        yaxis_title="Attacker Payoff",
        height=380,
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_comparison_bar(
    nash_def: float, nash_atk: float,
    lp_def: float,   lp_atk: float,
) -> go.Figure:
    """
    Grouped bar chart comparing Nash vs LP payoffs for both players.

    Args:
        nash_def, nash_atk: Nash equilibrium payoffs.
        lp_def,   lp_atk:   LP-optimal payoffs.

    Returns:
        Plotly Figure.
    """
    categories = ["Defender", "Attacker"]
    nash_vals  = [nash_def, nash_atk]
    lp_vals    = [lp_def,   lp_atk]

    fig = go.Figure(data=[
        go.Bar(
            name="Nash Equilibrium",
            x=categories,
            y=nash_vals,
            marker_color=DEFENDER_COLOR,
            text=[f"{v:.3f}" for v in nash_vals],
            textposition="outside",
        ),
        go.Bar(
            name="LP Optimal",
            x=categories,
            y=lp_vals,
            marker_color=ACCENT_COLOR,
            text=[f"{v:.3f}" for v in lp_vals],
            textposition="outside",
        ),
    ])

    fig.update_layout(
        title="Nash Equilibrium vs. Linear Programming Comparison",
        yaxis_title="Expected Payoff",
        barmode="group",
        height=350,
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_q_table(
    q_table: np.ndarray,
    defender_labels: list,
    attacker_labels: list,
) -> go.Figure:
    """
    Heatmap of the Q-learning agent's Q-table.

    Rows = states (attacker strategies), columns = actions (defender strategies).

    Args:
        q_table:         np.ndarray (n_states x n_actions).
        defender_labels: Column labels.
        attacker_labels: Row labels.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(go.Heatmap(
        z=q_table,
        x=defender_labels,
        y=attacker_labels,
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(q_table, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="State (Atk): %{y}<br>Action (Def): %{x}<br>Q-Value: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="Q-Table: Defender Q-Values (State × Action)",
        xaxis_title="Defender Action",
        yaxis_title="Last Attacker Move (State)",
        height=320,
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_attack_proba(proba: np.ndarray, strategy_names: list) -> go.Figure:
    """
    Horizontal bar chart of predicted attacker strategy probabilities.

    Args:
        proba:          Probability distribution (np.ndarray).
        strategy_names: Labels for each strategy.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(go.Bar(
        x=proba,
        y=strategy_names,
        orientation="h",
        marker=dict(
            color=proba,
            colorscale="Reds",
            showscale=False,
        ),
        text=[f"{p:.1%}" for p in proba],
        textposition="outside",
    ))

    fig.update_layout(
        title="Predicted Next Attacker Strategy",
        xaxis_title="Probability",
        height=280,
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(range=[0, 1.15])
    return fig


def plot_reward_history(reward_history: list) -> go.Figure:
    """
    Line chart of Q-learner reward per round (moving average included).

    Args:
        reward_history: List of per-round rewards.

    Returns:
        Plotly Figure.
    """
    rounds = list(range(1, len(reward_history) + 1))
    rewards = reward_history

    # Compute moving average (window=10)
    window = min(10, len(rewards))
    ma = pd.Series(rewards).rolling(window, min_periods=1).mean().tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=rewards,
        name="Per-Round Reward",
        line=dict(color=f"rgba(74,144,217,0.35)", width=1),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=ma,
        name=f"Moving Avg ({window})",
        line=dict(color=DEFENDER_COLOR, width=2.5),
        mode="lines",
    ))

    fig.update_layout(
        title="Q-Learning Defender: Reward Over Training",
        xaxis_title="Round",
        yaxis_title="Reward (Defender Payoff)",
        height=300,
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_payoff_matrix_heatmap(A: np.ndarray, B: np.ndarray,
                                def_labels: list, atk_labels: list) -> go.Figure:
    """
    Side-by-side heatmaps of both payoff matrices.

    Returns:
        Plotly Figure with two subplots.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Defender Payoffs (A)", "Attacker Payoffs (B)"),
        horizontal_spacing=0.12,
    )

    def make_heatmap(matrix, colorscale, row, col):
        fig.add_trace(go.Heatmap(
            z=matrix,
            x=atk_labels,
            y=def_labels,
            colorscale=colorscale,
            zmid=0,
            text=np.round(matrix, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            showscale=True,
        ), row=row, col=col)

    make_heatmap(A, "Blues",  1, 1)
    make_heatmap(B, "Reds",   1, 2)

    fig.update_layout(
        height=320,
        **_PLOTLY_LAYOUT,
    )
    return fig
