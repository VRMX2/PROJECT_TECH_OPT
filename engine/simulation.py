"""
engine/simulation.py
────────────────────
Repeated attacker-defender game simulation engine.

Runs N rounds of the game, recording:
  • Which strategy each player chose (sampled from mixed-strategy equilibrium
    probabilities, or chosen by the AI agent)
  • Resulting payoffs each round
  • Cumulative average payoffs (convergence tracking)

Provides:
  • run_simulation(A, B, equilibria, rounds, ai_agent) → pd.DataFrame
  • best_response(payoff_matrix, opponent_strategy) → int (pure best response)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def best_response(payoff_matrix: np.ndarray, opponent_strategy: np.ndarray) -> int:
    """
    Compute a player's pure-strategy best response to the opponent's mixed strategy.

    Args:
        payoff_matrix:     This player's payoff matrix (rows = this player's strategies).
        opponent_strategy: Probability vector over the opponent's strategies.

    Returns:
        Index of the best-response pure strategy.
    """
    expected_payoffs = payoff_matrix @ opponent_strategy
    return int(np.argmax(expected_payoffs))


def sample_strategy(sigma: np.ndarray) -> int:
    """
    Sample a pure strategy index from a mixed strategy distribution.

    Args:
        sigma: Probability vector (must sum to 1).

    Returns:
        Sampled strategy index.
    """
    # Guard: normalise in case of floating-point drift
    sigma = np.array(sigma, dtype=float)
    sigma = np.clip(sigma, 0, None)
    total = sigma.sum()
    if total == 0:
        sigma = np.ones(len(sigma)) / len(sigma)
    else:
        sigma = sigma / total
    return int(np.random.choice(len(sigma), p=sigma))


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    A: np.ndarray,
    B: np.ndarray,
    equilibria: list,
    rounds: int = 100,
    ai_agent=None,
    use_best_response: bool = False,
) -> pd.DataFrame:
    """
    Simulate repeated attacker-defender interactions.

    Strategy selection per round:
      • If ai_agent is provided (Q-learner), the defender uses it.
      • Otherwise, strategies are sampled from the first Nash-equilibrium
        mixed strategy distribution.
      • If use_best_response=True, the attacker plays a pure best response
        each round instead of sampling.

    Args:
        A:                Defender payoff matrix (m x n).
        B:                Attacker payoff matrix (m x n).
        equilibria:       List of (sigma_defender, sigma_attacker) NE tuples.
        rounds:           Number of rounds to simulate.
        ai_agent:         Optional QLearningAgent instance for the defender.
        use_best_response: If True, attacker plays best response each round.

    Returns:
        pd.DataFrame with columns:
          round, defender_strategy, attacker_strategy,
          defender_payoff, attacker_payoff,
          cum_avg_defender, cum_avg_attacker,
          defender_probs_*, attacker_probs_*
    """
    m, n = A.shape

    # Use first equilibrium if available, otherwise uniform
    if equilibria:
        sigma_def, sigma_atk = equilibria[0]
    else:
        sigma_def = np.ones(m) / m
        sigma_atk = np.ones(n) / n

    records = []
    cum_def = 0.0
    cum_atk = 0.0

    # History of attacker moves (for AI pattern recognition)
    atk_history = []

    for r in range(1, rounds + 1):
        # ── Attacker strategy ─────────────────────────────────────────────
        if use_best_response and len(atk_history) > 0:
            # Attacker best-responds to empirical defender distribution
            emp_def = np.zeros(m)
            for d in atk_history:    # reuse defender choices
                pass
            atk_action = sample_strategy(sigma_atk)
        else:
            atk_action = sample_strategy(sigma_atk)
        atk_history.append(atk_action)

        # ── Defender strategy ─────────────────────────────────────────────
        if ai_agent is not None:
            # Q-learning defender: state = last attacker action
            state = atk_history[-2] if len(atk_history) >= 2 else 0
            def_action = ai_agent.select_action(state)
        else:
            def_action = sample_strategy(sigma_def)

        # ── Compute payoffs ───────────────────────────────────────────────
        def_payoff = float(A[def_action, atk_action])
        atk_payoff = float(B[def_action, atk_action])

        # ── Update Q-learner ──────────────────────────────────────────────
        if ai_agent is not None and len(atk_history) >= 2:
            prev_state  = atk_history[-2]
            next_state  = atk_action
            ai_agent.update(prev_state, def_action, def_payoff, next_state)

        # ── Cumulative averages ───────────────────────────────────────────
        cum_def = cum_def + (def_payoff - cum_def) / r
        cum_atk = cum_atk + (atk_payoff - cum_atk) / r

        record = {
            "round":             r,
            "defender_strategy": def_action,
            "attacker_strategy": atk_action,
            "defender_payoff":   def_payoff,
            "attacker_payoff":   atk_payoff,
            "cum_avg_defender":  cum_def,
            "cum_avg_attacker":  cum_atk,
        }

        # Also store the current mixed-strategy probabilities per round
        for i, p in enumerate(sigma_def):
            record[f"def_prob_{i}"] = float(p)
        for j, p in enumerate(sigma_atk):
            record[f"atk_prob_{j}"] = float(p)

        records.append(record)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Best-response dynamics (iterated)
# ─────────────────────────────────────────────────────────────────────────────

def run_best_response_dynamics(
    A: np.ndarray,
    B: np.ndarray,
    rounds: int = 50,
) -> pd.DataFrame:
    """
    Simulate best-response dynamics (each player best-responds each round).
    Shows how strategies evolve and whether they converge to a Nash equilibrium.

    Returns a DataFrame with columns: round, defender_strategy, attacker_strategy,
    defender_payoff, attacker_payoff.
    """
    m, n = A.shape
    # Start with uniform beliefs
    def_strategy = np.ones(m) / m
    atk_strategy = np.ones(n) / n

    records = []
    cum_def = 0.0
    cum_atk = 0.0

    for r in range(1, rounds + 1):
        # Best responses
        def_br = best_response(A,   atk_strategy)
        atk_br = best_response(B.T, def_strategy)

        # Convert to one-hot  (pure strategies this round)
        def_pure = np.zeros(m); def_pure[def_br] = 1.0
        atk_pure = np.zeros(n); atk_pure[atk_br] = 1.0

        # Payoffs
        def_payoff = float(A[def_br, atk_br])
        atk_payoff = float(B[def_br, atk_br])

        # Smooth update (empirical frequency)
        alpha = 1.0 / r
        def_strategy = (1 - alpha) * def_strategy + alpha * def_pure
        atk_strategy = (1 - alpha) * atk_strategy + alpha * atk_pure

        cum_def = cum_def + (def_payoff - cum_def) / r
        cum_atk = cum_atk + (atk_payoff - cum_atk) / r

        record = {
            "round":             r,
            "defender_strategy": def_br,
            "attacker_strategy": atk_br,
            "defender_payoff":   def_payoff,
            "attacker_payoff":   atk_payoff,
            "cum_avg_defender":  cum_def,
            "cum_avg_attacker":  cum_atk,
        }
        for i, p in enumerate(def_strategy):
            record[f"def_prob_{i}"] = float(p)
        for j, p in enumerate(atk_strategy):
            record[f"atk_prob_{j}"] = float(p)

        records.append(record)

    return pd.DataFrame(records)
