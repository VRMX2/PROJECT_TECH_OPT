"""
tests/test_optimization.py
──────────────────────────
Unit tests for engine/optimization.py.

Tests cover:
  • solve_lp_defender()   – strategy sums to 1, value is finite, feasibility
  • solve_lp_attacker()   – mirrors defender, strategy length matches columns
  • compare_nash_vs_lp()  – output keys, efficiency gap direction, zero-sum identity
"""

import pytest
import numpy as np
from engine.optimization import (
    solve_lp_defender,
    solve_lp_attacker,
    compare_nash_vs_lp,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def matching_pennies():
    """Zero-sum 2×2 (minimax value = 0)."""
    A = np.array([[ 1.0, -1.0],
                  [-1.0,  1.0]])
    return A


@pytest.fixture
def pd_matrices():
    """Prisoner's Dilemma payoff matrices (general-sum)."""
    A = np.array([[3.0, 0.0],
                  [5.0, 1.0]])
    B = np.array([[3.0, 5.0],
                  [0.0, 1.0]])
    return A, B


@pytest.fixture
def asymmetric_3x4():
    """Non-square 3×4 defender payoff matrix."""
    return np.array([[2.0, 1.0, 3.0, 0.0],
                     [0.0, 4.0, 1.0, 2.0],
                     [3.0, 0.0, 2.0, 1.0]])


# ─────────────────────────────────────────────────────────────────────────────
# solve_lp_defender
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveLPDefender:

    def test_returns_dict_with_required_keys(self, matching_pennies):
        result = solve_lp_defender(matching_pennies)
        assert "strategy" in result
        assert "value" in result
        assert "success" in result
        assert "message" in result

    def test_success_flag_true_for_feasible(self, matching_pennies):
        result = solve_lp_defender(matching_pennies)
        assert result["success"] is True

    def test_strategy_sums_to_one(self, matching_pennies):
        result = solve_lp_defender(matching_pennies)
        assert abs(result["strategy"].sum() - 1.0) < 1e-6

    def test_strategy_is_non_negative(self, matching_pennies):
        result = solve_lp_defender(matching_pennies)
        assert (result["strategy"] >= -1e-9).all()

    def test_strategy_length_matches_rows(self, matching_pennies):
        result = solve_lp_defender(matching_pennies)
        assert len(result["strategy"]) == matching_pennies.shape[0]

    def test_zero_sum_minimax_value_near_zero(self, matching_pennies):
        """Matching Pennies minimax value should be ≈ 0."""
        result = solve_lp_defender(matching_pennies)
        assert abs(result["value"]) < 1e-4

    def test_zero_sum_mixed_strategy_near_uniform(self, matching_pennies):
        """Optimal defender strategy for Matching Pennies is (0.5, 0.5)."""
        result = solve_lp_defender(matching_pennies)
        np.testing.assert_allclose(result["strategy"], [0.5, 0.5], atol=1e-4)

    def test_asymmetric_game_strategy_correct_length(self, asymmetric_3x4):
        result = solve_lp_defender(asymmetric_3x4)
        assert len(result["strategy"]) == 3  # 3 defender strategies (rows)

    def test_asymmetric_game_strategy_sums_to_one(self, asymmetric_3x4):
        result = solve_lp_defender(asymmetric_3x4)
        assert abs(result["strategy"].sum() - 1.0) < 1e-6

    def test_dominant_strategy_game(self):
        """When defender has a dominant strategy, LP should concentrate on it."""
        # Strategy 0 always gives ≥ 10, strategy 1 always gives ≤ 0
        A = np.array([[10.0, 10.0],
                      [ 0.0,  0.0]])
        result = solve_lp_defender(A)
        assert result["success"] is True
        # Strategy weight on row 0 should be ≈ 1.0
        assert result["strategy"][0] > 0.9

    def test_value_is_float(self, matching_pennies):
        result = solve_lp_defender(matching_pennies)
        assert isinstance(result["value"], float)

    def test_1x1_game(self):
        A = np.array([[7.0]])
        result = solve_lp_defender(A)
        assert result["success"] is True
        np.testing.assert_allclose(result["strategy"], [1.0], atol=1e-6)
        assert abs(result["value"] - 7.0) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# solve_lp_attacker
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveLPAttacker:

    def test_returns_dict_with_required_keys(self, pd_matrices):
        A, B = pd_matrices
        result = solve_lp_attacker(B)
        assert "strategy" in result
        assert "value" in result
        assert "success" in result
        assert "message" in result

    def test_success_flag_true(self, pd_matrices):
        A, B = pd_matrices
        result = solve_lp_attacker(B)
        assert result["success"] is True

    def test_strategy_sums_to_one(self, pd_matrices):
        A, B = pd_matrices
        result = solve_lp_attacker(B)
        assert abs(result["strategy"].sum() - 1.0) < 1e-6

    def test_strategy_is_non_negative(self, pd_matrices):
        A, B = pd_matrices
        result = solve_lp_attacker(B)
        assert (result["strategy"] >= -1e-9).all()

    def test_strategy_length_matches_columns(self, pd_matrices):
        A, B = pd_matrices
        result = solve_lp_attacker(B)
        # B is 2×2, attacker has 2 strategies (columns)
        assert len(result["strategy"]) == B.shape[1]

    def test_asymmetric_attacker_strategy_length(self, asymmetric_3x4):
        # Treat 3×4 as attacker matrix: attacker has 4 column strategies
        result = solve_lp_attacker(asymmetric_3x4)
        assert len(result["strategy"]) == 4

    def test_zero_sum_minimax_value_near_zero(self, matching_pennies):
        """For Matching Pennies (B = -A), attacker minimax value ≈ 0."""
        B = -matching_pennies
        result = solve_lp_attacker(B)
        assert abs(result["value"]) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# compare_nash_vs_lp
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareNashVsLP:

    def test_returns_dict(self, pd_matrices):
        A, B = pd_matrices
        result = compare_nash_vs_lp(A, B)
        assert isinstance(result, dict)

    def test_required_keys_no_nash_provided(self, pd_matrices):
        A, B = pd_matrices
        result = compare_nash_vs_lp(A, B)
        assert "lp_defender_strategy" in result
        assert "lp_defender_value" in result
        assert "lp_defender_success" in result
        assert "lp_attacker_strategy" in result
        assert "lp_attacker_value" in result
        assert "lp_attacker_success" in result

    def test_nash_keys_absent_when_not_provided(self, pd_matrices):
        A, B = pd_matrices
        result = compare_nash_vs_lp(A, B)
        assert "nash_defender_payoff" not in result
        assert "nash_attacker_payoff" not in result

    def test_efficiency_gap_keys_present_when_nash_provided(self, pd_matrices):
        A, B = pd_matrices
        result = compare_nash_vs_lp(A, B, nash_def_payoff=1.0, nash_atk_payoff=1.0)
        assert "nash_defender_payoff" in result
        assert "nash_attacker_payoff" in result
        assert "defender_efficiency_gain" in result
        assert "attacker_efficiency_gain" in result

    def test_efficiency_gain_is_numeric_or_none(self, pd_matrices):
        A, B = pd_matrices
        result = compare_nash_vs_lp(A, B, nash_def_payoff=2.0, nash_atk_payoff=2.0)
        gain_def = result["defender_efficiency_gain"]
        gain_atk = result["attacker_efficiency_gain"]
        # Must be a float or None
        assert gain_def is None or isinstance(gain_def, float)
        assert gain_atk is None or isinstance(gain_atk, float)

    def test_zero_sum_lp_values_opposite_sign(self, matching_pennies):
        """In a zero-sum game, LP values for defender and attacker are negatives."""
        A = matching_pennies
        B = -A
        result = compare_nash_vs_lp(A, B)
        if result["lp_defender_success"] and result["lp_attacker_success"]:
            np.testing.assert_allclose(
                result["lp_defender_value"] + result["lp_attacker_value"],
                0.0, atol=1e-4,
                err_msg="Zero-sum: defender value + attacker value should equal 0"
            )

    def test_strategies_are_ndarrays(self, pd_matrices):
        A, B = pd_matrices
        result = compare_nash_vs_lp(A, B)
        assert isinstance(result["lp_defender_strategy"], np.ndarray)
        assert isinstance(result["lp_attacker_strategy"], np.ndarray)

    def test_strategies_valid_probability_vectors(self, pd_matrices):
        A, B = pd_matrices
        result = compare_nash_vs_lp(A, B)
        def_strat = result["lp_defender_strategy"]
        atk_strat = result["lp_attacker_strategy"]
        assert abs(def_strat.sum() - 1.0) < 1e-6
        assert abs(atk_strat.sum() - 1.0) < 1e-6
        assert (def_strat >= -1e-9).all()
        assert (atk_strat >= -1e-9).all()

    def test_only_defender_nash_payoff_provided(self, pd_matrices):
        A, B = pd_matrices
        result = compare_nash_vs_lp(A, B, nash_def_payoff=2.5)
        assert "nash_defender_payoff" in result
        assert "defender_efficiency_gain" in result
        assert "nash_attacker_payoff" not in result
        assert "attacker_efficiency_gain" not in result
