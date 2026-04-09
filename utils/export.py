"""
utils/export.py
───────────────
CSV export utilities for simulation results and game configurations.

Provides:
  • simulation_to_csv(df)             → CSV bytes for download
  • pareto_to_csv(pareto_outcomes)    → CSV bytes for download
  • equilibria_to_csv(classified_eqs) → CSV bytes for download
  • config_to_csv(A, B, strategies)   → CSV bytes for download
"""

import numpy as np
import pandas as pd
import io
import json


def simulation_to_csv(df: pd.DataFrame) -> bytes:
    """
    Serialise a simulation results DataFrame to CSV bytes.

    Args:
        df: DataFrame returned by engine.simulation.run_simulation().

    Returns:
        UTF-8 encoded CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def pareto_to_csv(pareto_outcomes: list) -> bytes:
    """
    Serialise Pareto-optimal outcomes to CSV bytes.

    Args:
        pareto_outcomes: List of dicts from engine.game_theory.compute_pareto_optimal().

    Returns:
        UTF-8 encoded CSV bytes.
    """
    if not pareto_outcomes:
        return b"row,col,defender_payoff,attacker_payoff\n"
    df = pd.DataFrame(pareto_outcomes)
    return df.to_csv(index=False).encode("utf-8")


def equilibria_to_csv(classified_eqs: list) -> bytes:
    """
    Serialise Nash equilibria to CSV bytes.

    Args:
        classified_eqs: List of dicts from engine.game_theory.classify_equilibria().

    Returns:
        UTF-8 encoded CSV bytes.
    """
    rows = []
    for i, eq in enumerate(classified_eqs):
        row = {"equilibrium_index": i, "type": eq["type"]}
        for j, p in enumerate(eq["sigma_defender"]):
            row[f"defender_strategy_{j}"] = round(float(p), 6)
        for j, p in enumerate(eq["sigma_attacker"]):
            row[f"attacker_strategy_{j}"] = round(float(p), 6)
        rows.append(row)

    if not rows:
        return b"equilibrium_index,type\n"
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def config_to_csv(A: np.ndarray, B: np.ndarray, defender_labels: list, attacker_labels: list) -> bytes:
    """
    Export both payoff matrices to a single CSV with labelled rows/columns.

    Args:
        A:               Defender payoff matrix.
        B:               Attacker payoff matrix.
        defender_labels: Row labels (defender strategies).
        attacker_labels: Column labels (attacker strategies).

    Returns:
        UTF-8 encoded CSV bytes.
    """
    buf = io.StringIO()
    buf.write("DEFENDER PAYOFF MATRIX (A)\n")
    df_A = pd.DataFrame(A, index=defender_labels, columns=attacker_labels)
    df_A.to_csv(buf)
    buf.write("\nATTACKER PAYOFF MATRIX (B)\n")
    df_B = pd.DataFrame(B, index=defender_labels, columns=attacker_labels)
    df_B.to_csv(buf)
    return buf.getvalue().encode("utf-8")


def export_scenario_json(A: np.ndarray, B: np.ndarray, def_labels: list, atk_labels: list) -> str:
    """
    Serialise a custom scenario into a formatted JSON string.
    """
    scenario = {
        "defender_labels": def_labels,
        "attacker_labels": atk_labels,
        "A": A.tolist(),
        "B": B.tolist()
    }
    return json.dumps(scenario, indent=2)


def parse_scenario_json(json_bytes: bytes) -> dict:
    """
    Parse a JSON scenario back into NumPy matrices and string labels.
    """
    scenario = json.loads(json_bytes.decode("utf-8"))
    return {
        "defender_labels": scenario["defender_labels"],
        "attacker_labels": scenario["attacker_labels"],
        "A": np.array(scenario["A"]),
        "B": np.array(scenario["B"])
    }
