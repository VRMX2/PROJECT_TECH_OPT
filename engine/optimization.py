"""
engine/optimization.py
──────────────────────
Centralized Linear Programming (LP) comparison module.

For a zero-sum game the LP formulation is the dual of the Nash minimax:
  • Defender (row player) maximizes min expected payoff
  • Attacker  (col player) maximizes min expected payoff (from their matrix)

Uses scipy.optimize.linprog to solve and compares results to Nash equilibrium.

Provides:
  • solve_lp_defender(A) → dict with strategy, value, success, message
  • solve_lp_attacker(B) → dict with strategy, value, success, message
  • compare_nash_vs_lp(A, B, nash_def_payoff, nash_atk_payoff) → dict
"""

import numpy as np
from scipy.optimize import linprog
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# LP Solver — Defender's perspective (row player)
# ─────────────────────────────────────────────────────────────────────────────

def solve_lp_defender(A: np.ndarray) -> dict:
    """
    Solve the Defender's minimax LP for a zero-sum game.

    Formulation (maximise game value v):
      max  v
      s.t. A^T x >= v * 1   (defender gets at least v vs. any attacker strategy)
           sum(x) = 1
           x >= 0

    Reformulated for linprog (minimise -v):
      Variables: [x_0 ... x_{m-1}, v]
      min  c^T y  where  c = [0,...,0,-1]
      s.t. -A^T x + v <= 0   (n inequality constraints)
           sum(x)     = 1    (equality)
           0 <= x_i
           v unconstrained

    Args:
        A: Defender payoff matrix (m rows = defender strategies,
                                   n cols = attacker strategies).

    Returns:
        dict with keys:
          strategy (np.ndarray), value (float), success (bool), message (str)
    """
    m, n = A.shape

    # Objective: minimise -v  (len = m+1 variables: x_0..x_{m-1}, v)
    c = np.zeros(m + 1)
    c[-1] = -1.0          # coefficient on v

    # Inequality constraints: -A[:,j]^T x + v <= 0  for each attacker strategy j
    A_ub = np.zeros((n, m + 1))
    for j in range(n):
        A_ub[j, :m] = -A[:, j]
        A_ub[j, -1] =  1.0
    b_ub = np.zeros(n)

    # Equality: sum(x) = 1  (v not counted)
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # Bounds: x_i in [0, 1], v unbounded
    bounds = [(0.0, 1.0)] * m + [(None, None)]

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs"
    )

    if result.success:
        strategy = np.clip(result.x[:m], 0.0, 1.0)
        total = strategy.sum()
        strategy = strategy / total if total > 1e-12 else np.ones(m) / m
        return {
            "strategy": strategy,
            "value":    float(result.x[-1]),
            "success":  True,
            "message":  "Optimal solution found",
        }
    else:
        return {
            "strategy": np.ones(m) / m,
            "value":    float("nan"),
            "success":  False,
            "message":  result.message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LP Solver — Attacker's perspective (column player)
# ─────────────────────────────────────────────────────────────────────────────

def solve_lp_attacker(B: np.ndarray) -> dict:
    """
    Solve the Attacker's minimax LP using payoff matrix B.

    The attacker is the column player. We treat -B^T as the "defender" payoff
    matrix in a transposed game, so the same solver applies.

    Args:
        B: Attacker payoff matrix (m x n).

    Returns:
        dict with keys: strategy (np.ndarray over n cols), value, success, message.
    """
    return solve_lp_defender(-B.T)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison: Nash Equilibrium vs LP Optimal
# ─────────────────────────────────────────────────────────────────────────────

def compare_nash_vs_lp(
    A: np.ndarray,
    B: np.ndarray,
    nash_def_payoff: Optional[float] = None,
    nash_atk_payoff: Optional[float] = None,
) -> dict:
    """
    Compare Nash Equilibrium payoffs with LP-optimal (minimax) payoffs.

    For zero-sum games the two coincide (minimax theorem).
    For general-sum games there may be an efficiency gap.

    Args:
        A:               Defender payoff matrix (m x n).
        B:               Attacker payoff matrix (m x n).
        nash_def_payoff: Defender expected payoff at Nash equilibrium (or None).
        nash_atk_payoff: Attacker expected payoff at Nash equilibrium (or None).

    Returns:
        dict with all comparison metrics including LP strategies and efficiency gaps.
    """
    lp_def = solve_lp_defender(A)
    lp_atk = solve_lp_attacker(B)

    result = {
        "lp_defender_strategy": lp_def["strategy"],
        "lp_defender_value":    lp_def["value"],
        "lp_defender_success":  lp_def["success"],
        "lp_attacker_strategy": lp_atk["strategy"],
        "lp_attacker_value":    lp_atk["value"],
        "lp_attacker_success":  lp_atk["success"],
    }

    if nash_def_payoff is not None:
        result["nash_defender_payoff"] = nash_def_payoff
        if lp_def["success"] and not np.isnan(lp_def["value"]):
            result["defender_efficiency_gain"] = lp_def["value"] - nash_def_payoff
        else:
            result["defender_efficiency_gain"] = None

    if nash_atk_payoff is not None:
        result["nash_attacker_payoff"] = nash_atk_payoff
        if lp_atk["success"] and not np.isnan(lp_atk["value"]):
            result["attacker_efficiency_gain"] = lp_atk["value"] - nash_atk_payoff
        else:
            result["attacker_efficiency_gain"] = None

    return result
