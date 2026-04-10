"""
tests/test_game_theory.py
─────────────────────────
Unit tests for engine/game_theory.py.

Tests cover:
  • create_game()              – correct nashpy Game construction
  • compute_nash_equilibria()  – pure and mixed NE detection, probabilities sum to 1
  • classify_equilibria()      – correct Pure/Mixed labelling
  • compute_pareto_optimal()   – identifies non-dominated outcomes
  • get_payoff_at()            – expected payoff calculation
"""

import pytest
import numpy as np
from engine.game_theory import (
    create_game,
    compute_nash_equilibria,
    classify_equilibria,
    compute_pareto_optimal,
    get_payoff_at,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared payoff matrices
# ─────────────────────────────────────────────────────────────────────────────

# Prisoner's Dilemma (general-sum, pure NE at (1,1) = Defect/Defect)
PD_A = np.array([[3.0, 0.0],
                 [5.0, 1.0]])
PD_B = np.array([[3.0, 5.0],
                 [0.0, 1.0]])

# Matching Pennies (zero-sum, unique mixed NE at (0.5, 0.5))
MP_A = np.array([[ 1.0, -1.0],
                 [-1.0,  1.0]])
MP_B = np.array([[-1.0,  1.0],
                 [ 1.0, -1.0]])

# Coordination game (two pure NE)
CG_A = np.array([[2.0, 0.0],
                 [0.0, 1.0]])
CG_B = np.array([[2.0, 0.0],
                 [0.0, 1.0]])


# ─────────────────────────────────────────────────────────────────────────────
# create_game
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateGame:

    def test_returns_nashpy_game(self):
        import nashpy as nash
        game = create_game(PD_A, PD_B)
        assert isinstance(game, nash.Game)

    def test_game_stores_correct_matrices(self):
        game = create_game(MP_A, MP_B)
        np.testing.assert_array_equal(game.payoff_matrices[0], MP_A)
        np.testing.assert_array_equal(game.payoff_matrices[1], MP_B)


# ─────────────────────────────────────────────────────────────────────────────
# compute_nash_equilibria
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeNashEquilibria:

    def test_returns_list(self):
        game = create_game(PD_A, PD_B)
        eq = compute_nash_equilibria(game)
        assert isinstance(eq, list)

    def test_prisoners_dilemma_has_one_ne(self):
        game = create_game(PD_A, PD_B)
        eq = compute_nash_equilibria(game)
        assert len(eq) == 1

    def test_coordination_game_has_multiple_ne(self):
        game = create_game(CG_A, CG_B)
        eq = compute_nash_equilibria(game)
        # At minimum both pure NE should be found
        assert len(eq) >= 2

    def test_ne_probabilities_sum_to_one(self):
        game = create_game(MP_A, MP_B)
        eq = compute_nash_equilibria(game)
        assert len(eq) > 0
        for sigma_r, sigma_c in eq:
            assert abs(sigma_r.sum() - 1.0) < 1e-6
            assert abs(sigma_c.sum() - 1.0) < 1e-6

    def test_matching_pennies_mixed_ne(self):
        game = create_game(MP_A, MP_B)
        eq = compute_nash_equilibria(game)
        assert len(eq) > 0
        sigma_r, sigma_c = eq[0]
        # Mixed NE should be near (0.5, 0.5)
        np.testing.assert_allclose(sigma_r, [0.5, 0.5], atol=1e-4)
        np.testing.assert_allclose(sigma_c, [0.5, 0.5], atol=1e-4)

    def test_ne_probabilities_non_negative(self):
        game = create_game(PD_A, PD_B)
        eq = compute_nash_equilibria(game)
        for sigma_r, sigma_c in eq:
            assert (sigma_r >= -1e-9).all()
            assert (sigma_c >= -1e-9).all()


# ─────────────────────────────────────────────────────────────────────────────
# classify_equilibria
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyEquilibria:

    def test_returns_list_of_dicts(self):
        game = create_game(PD_A, PD_B)
        eq   = compute_nash_equilibria(game)
        classified = classify_equilibria(eq)
        assert isinstance(classified, list)
        for item in classified:
            assert "sigma_defender" in item
            assert "sigma_attacker" in item
            assert "type" in item

    def test_prisoners_dilemma_ne_classified_pure(self):
        game = create_game(PD_A, PD_B)
        eq   = compute_nash_equilibria(game)
        classified = classify_equilibria(eq)
        assert classified[0]["type"] == "Pure"

    def test_matching_pennies_ne_classified_mixed(self):
        game = create_game(MP_A, MP_B)
        eq   = compute_nash_equilibria(game)
        classified = classify_equilibria(eq)
        assert classified[0]["type"] == "Mixed"

    def test_empty_equilibria_returns_empty_list(self):
        result = classify_equilibria([])
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# compute_pareto_optimal
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeParetoOptimal:

    def test_returns_list_of_dicts(self):
        pareto = compute_pareto_optimal(PD_A, PD_B)
        assert isinstance(pareto, list)
        assert len(pareto) > 0
        for item in pareto:
            assert "row" in item
            assert "col" in item
            assert "defender_payoff" in item
            assert "attacker_payoff" in item

    def test_dominated_outcome_excluded(self):
        # (1,1) in PD is (1,1) payoffs — dominated by (0,0)=(3,3)
        pareto = compute_pareto_optimal(PD_A, PD_B)
        dominated = [(p["row"], p["col"]) for p in pareto if
                     p["defender_payoff"] == 1.0 and p["attacker_payoff"] == 1.0]
        assert len(dominated) == 0, "Dominated (Defect,Defect) should not be Pareto-optimal"

    def test_cooperative_outcome_in_pareto(self):
        # (0,0) = (3,3) in PD — should be Pareto-optimal
        pareto = compute_pareto_optimal(PD_A, PD_B)
        coop = [(p["row"], p["col"]) for p in pareto if
                p["defender_payoff"] == 3.0 and p["attacker_payoff"] == 3.0]
        assert len(coop) == 1

    def test_pareto_outcomes_all_non_dominated(self):
        """No Pareto-optimal outcome should strictly dominate another."""
        pareto = compute_pareto_optimal(CG_A, CG_B)
        for i, p1 in enumerate(pareto):
            for j, p2 in enumerate(pareto):
                if i == j:
                    continue
                # p2 should not strictly dominate p1
                assert not (
                    p2["defender_payoff"] >= p1["defender_payoff"]
                    and p2["attacker_payoff"] >= p1["attacker_payoff"]
                    and (p2["defender_payoff"] > p1["defender_payoff"]
                         or p2["attacker_payoff"] > p1["attacker_payoff"])
                ), f"Outcome {p2} dominates {p1} — should not both be Pareto-optimal"

    def test_single_outcome_game(self):
        A = np.array([[5.0]])
        B = np.array([[3.0]])
        pareto = compute_pareto_optimal(A, B)
        assert len(pareto) == 1
        assert pareto[0]["defender_payoff"] == 5.0
        assert pareto[0]["attacker_payoff"] == 3.0


# ─────────────────────────────────────────────────────────────────────────────
# get_payoff_at
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPayoffAt:

    def test_pure_strategy_payoff(self):
        game = create_game(PD_A, PD_B)
        # Both play strategy 1 (Defect) → (1, 1)
        sigma_r = np.array([0.0, 1.0])
        sigma_c = np.array([0.0, 1.0])
        def_pay, atk_pay = get_payoff_at(game, sigma_r, sigma_c)
        assert abs(def_pay - 1.0) < 1e-9
        assert abs(atk_pay - 1.0) < 1e-9

    def test_cooperative_pure_strategy_payoff(self):
        game = create_game(PD_A, PD_B)
        # Both cooperate (strategy 0) → (3, 3)
        sigma_r = np.array([1.0, 0.0])
        sigma_c = np.array([1.0, 0.0])
        def_pay, atk_pay = get_payoff_at(game, sigma_r, sigma_c)
        assert abs(def_pay - 3.0) < 1e-9
        assert abs(atk_pay - 3.0) < 1e-9

    def test_mixed_strategy_payoff_matching_pennies(self):
        game = create_game(MP_A, MP_B)
        # At (0.5,0.5) NE, expected payoff for both is 0
        sigma = np.array([0.5, 0.5])
        def_pay, atk_pay = get_payoff_at(game, sigma, sigma)
        assert abs(def_pay - 0.0) < 1e-9
        assert abs(atk_pay - 0.0) < 1e-9

    def test_returns_floats(self):
        game = create_game(PD_A, PD_B)
        sigma_r = np.array([0.5, 0.5])
        sigma_c = np.array([0.5, 0.5])
        def_pay, atk_pay = get_payoff_at(game, sigma_r, sigma_c)
        assert isinstance(def_pay, float)
        assert isinstance(atk_pay, float)
