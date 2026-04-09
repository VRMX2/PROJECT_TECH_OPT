"""
ui/matrix_editor.py
────────────────────
Streamlit widgets for editing payoff matrices and game configuration.

Provides:
  • render_matrix_editor(label, matrix, key) → edited np.ndarray
  • render_strategy_labels(n, prefix) → list of strategy name strings
  • build_default_matrices(m, n, scenario)  → (A, B) default payoff matrices
"""

import numpy as np
import streamlit as st
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Scenario presets
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_PRESETS = {
    "Zero-Sum (Classic)": {
        "description": "Attacker gains equal what defender loses. Strictly competitive.",
        "A": np.array([[-5.0, 3.0], [2.0, -1.0]]),
        "B": np.array([[5.0, -3.0], [-2.0, 1.0]]),
        "def_labels": ["Patch System", "Monitor Network"],
        "atk_labels": ["Exploit Vuln", "Phishing"],
    },
    "General-Sum (Realistic)": {
        "description": "Both players have independent payoffs including security costs.",
        "A": np.array([[-2.0, -8.0, -1.0],
                       [-4.0, -3.0, -6.0],
                       [-1.0, -5.0, -2.0]]),
        "B": np.array([[8.0, 3.0, 2.0],
                       [5.0, 7.0, 1.0],
                       [4.0, 6.0, 9.0]]),
        "def_labels": ["Firewall Rules", "Honeypot Deploy", "Full Lockdown"],
        "atk_labels": ["Port Scan", "Malware Drop", "Social Engineer"],
    },
    "Asymmetric (Advanced Persistent Threat)": {
        "description": "Attacker is patient; small gains from each vector accumulate.",
        "A": np.array([[-10.0, -2.0, -6.0, -1.0],
                       [-3.0,  -8.0, -2.0, -9.0],
                       [-1.0,  -4.0, -7.0, -3.0],
                       [-5.0,  -1.0, -3.0, -4.0]]),
        "B": np.array([[9.0, 2.0, 6.0, 1.0],
                       [3.0, 7.0, 2.0, 8.0],
                       [1.0, 4.0, 7.0, 3.0],
                       [5.0, 1.0, 4.0, 4.0]]),
        "def_labels": ["IDS Active", "Patch All", "Isolate Segment", "Full Response"],
        "atk_labels": ["Recon", "Lateral Move", "Data Exfil", "Ransomware"],
    },
}


def build_default_matrices(m: int, n: int) -> tuple:
    """
    Build default (m x n) payoff matrices for a cybersecurity game.

    Defender payoffs (A): negative values (security costs/losses).
    Attacker payoffs (B): positive for successful attacks.

    Args:
        m: Number of defender strategies (rows).
        n: Number of attacker strategies (columns).

    Returns:
        Tuple (A, B) of np.ndarray with shape (m, n).
    """
    rng = np.random.default_rng(seed=0)
    A = -rng.uniform(1, 10, size=(m, n)).round(1)   # defender loses on attack
    B =  rng.uniform(1, 10, size=(m, n)).round(1)   # attacker gains
    return A, B


def render_matrix_editor(
    label: str,
    matrix: np.ndarray,
    row_labels: list,
    col_labels: list,
    key: str,
) -> np.ndarray:
    """
    Render an editable payoff matrix using st.data_editor.

    Args:
        label:      Section label shown to the user.
        matrix:     Current matrix values (m x n np.ndarray).
        row_labels: Row header labels (defender strategies).
        col_labels: Column header labels (attacker strategies).
        key:        Unique Streamlit widget key.

    Returns:
        Edited matrix as np.ndarray.
    """
    st.markdown(f"**{label}**")

    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)

    edited_df = st.data_editor(
        df,
        key=key,
        use_container_width=True,
        num_rows="fixed",
    )

    return edited_df.to_numpy(dtype=float)


def render_strategy_config(prefix: str, default_n: int, key: str) -> tuple:
    """
    Render a slider for number of strategies and text inputs for strategy names.

    Args:
        prefix:    "Defender" or "Attacker".
        default_n: Default number of strategies.
        key:       Unique key prefix.

    Returns:
        (n_strategies, strategy_labels) tuple.
    """
    n = st.slider(
        f"Number of {prefix} Strategies",
        min_value=2, max_value=6,
        value=default_n,
        key=f"{key}_n",
    )

    labels = []
    for i in range(n):
        default = f"{prefix[0]}{i+1}"
        lbl = st.text_input(
            f"{prefix} Strategy {i+1} name",
            value=default,
            key=f"{key}_label_{i}",
            label_visibility="collapsed",
        )
        labels.append(lbl or default)

    return n, labels
