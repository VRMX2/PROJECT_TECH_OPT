"""
engine/game_theory.py
─────────────────────
Game-theoretic engine using nashpy.

Provides:
  • create_game(A, B)            → nashpy.Game
  • compute_nash_equilibria(game) → list of (sigma_r, sigma_c) tuples
  • compute_pareto_optimal(A, B) → list of (i, j) index pairs that are Pareto-optimal
  • get_payoff_at(A, B, eq)      → (defender_payoff, attacker_payoff) at an equilibrium
"""

import numpy as np
import nashpy as nash
from itertools import product


# ─────────────────────────────────────────────────────────────────────────────
# Game Creation
# ─────────────────────────────────────────────────────────────────────────────

def create_game(A: np.ndarray, B: np.ndarray) -> nash.Game:
    """
    Create a two-player general-sum game.

    Args:
        A: Payoff matrix for the Defender (row player). Shape (m, n).
        B: Payoff matrix for the Attacker (column player). Shape (m, n).

    Returns:
        A nashpy Game object.
    """
    return nash.Game(A, B)


# ─────────────────────────────────────────────────────────────────────────────
# Nash Equilibria
# ─────────────────────────────────────────────────────────────────────────────

def compute_nash_equilibria(game: nash.Game) -> list:
    """
    Compute all Nash Equilibria (pure and mixed) using support enumeration.

    Args:
        game: A nashpy Game object.

    Returns:
        List of (sigma_defender, sigma_attacker) tuples, where each sigma is
        a numpy array of mixed strategy probabilities.
        Empty list if none found.
    """
    equilibria = []
    try:
        for eq in game.support_enumeration():
            equilibria.append(eq)
    except Exception as e:
        print(f"[game_theory] Nash computation error: {e}")
    return equilibria


def classify_equilibria(equilibria: list) -> list:
    """
    Classify each Nash Equilibrium as 'Pure' or 'Mixed'.

    Args:
        equilibria: List of (sigma_r, sigma_c) tuples.

    Returns:
        List of dicts with keys: sigma_defender, sigma_attacker, type.
    """
    classified = []
    for sigma_r, sigma_c in equilibria:
        eq_type = "Pure" if (
            np.max(sigma_r) == 1.0 and np.max(sigma_c) == 1.0
        ) else "Mixed"
        classified.append({
            "sigma_defender": sigma_r,
            "sigma_attacker": sigma_c,
            "type": eq_type,
        })
    return classified


# ─────────────────────────────────────────────────────────────────────────────
# Pareto Optimality
# ─────────────────────────────────────────────────────────────────────────────

def compute_pareto_optimal(A: np.ndarray, B: np.ndarray) -> list:
    """
    Identify Pareto-optimal pure-strategy outcomes.

    An outcome (i, j) is Pareto-optimal if no other outcome (i', j') gives
    both players simultaneously higher (or equal) payoffs.

    Args:
        A: Defender payoff matrix (m x n).
        B: Attacker payoff matrix (m x n).

    Returns:
        List of dicts: {row, col, defender_payoff, attacker_payoff}
    """
    m, n = A.shape
    all_outcomes = [(i, j, A[i, j], B[i, j]) for i, j in product(range(m), range(n))]
    pareto = []

    for (i, j, a_ij, b_ij) in all_outcomes:
        dominated = False
        for (i2, j2, a2, b2) in all_outcomes:
            if (i2, j2) == (i, j):
                continue
            # (i2,j2) dominates (i,j) if both payoffs are ≥ and at least one is >
            if a2 >= a_ij and b2 >= b_ij and (a2 > a_ij or b2 > b_ij):
                dominated = True
                break
        if not dominated:
            pareto.append({
                "row": i,
                "col": j,
                "defender_payoff": float(a_ij),
                "attacker_payoff": float(b_ij),
            })

    return pareto


# ─────────────────────────────────────────────────────────────────────────────
# Payoff Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def get_payoff_at(game: nash.Game, sigma_r: np.ndarray, sigma_c: np.ndarray) -> tuple:
    """
    Compute expected payoffs at a given strategy profile.

    Args:
        game:    nashpy Game object.
        sigma_r: Defender mixed strategy (array of probabilities).
        sigma_c: Attacker mixed strategy (array of probabilities).

    Returns:
        (defender_expected_payoff, attacker_expected_payoff) as floats.
    """
    payoffs = game[sigma_r, sigma_c]
    return float(payoffs[0]), float(payoffs[1])
