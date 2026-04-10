"""
tests/test_simulation.py
─────────────────────────
Unit tests for engine/simulation.py.

Tests cover:
  • best_response()             – returns index of highest expected-payoff strategy
  • sample_strategy()           – valid index, handles edge cases
  • run_simulation()            – DataFrame shape, columns, cumulative averages
  • run_best_response_dynamics() – convergence shape, monotone cumulative average
"""

import pytest
import numpy as np
import pandas as pd
from engine.simulation import (
    best_response,
    sample_strategy,
    run_simulation,
    run_best_response_dynamics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def pd_matrices():
    """Prisoner's Dilemma payoff matrices (2×2)."""
    A = np.array([[3.0, 0.0],
                  [5.0, 1.0]])
    B = np.array([[3.0, 5.0],
                  [0.0, 1.0]])
    return A, B


@pytest.fixture
def uniform_equilibria(pd_matrices):
    """Uniform mixed strategy equilibrium (used as simple test fixture)."""
    A, B = pd_matrices
    sigma = np.array([0.5, 0.5])
    return [(sigma.copy(), sigma.copy())]


@pytest.fixture
def pure_equilibria():
    """Pure NE: defender plays 1, attacker plays 1 (both deterministic)."""
    return [(np.array([0.0, 1.0]), np.array([0.0, 1.0]))]


# ─────────────────────────────────────────────────────────────────────────────
# best_response
# ─────────────────────────────────────────────────────────────────────────────

class TestBestResponse:

    def test_dominant_pure_strategy(self):
        # Strategy 1 always gives higher payoff whatever opponent does
        A = np.array([[0.0, 0.0],
                      [2.0, 2.0]])
        opponent = np.array([0.5, 0.5])
        assert best_response(A, opponent) == 1

    def test_returns_integer(self):
        A = np.array([[1.0, 2.0],
                      [3.0, 0.0]])
        opponent = np.array([0.5, 0.5])
        result = best_response(A, opponent)
        assert isinstance(result, int)

    def test_pure_opponent_strategy(self):
        # Opponent always plays col 0 → compare A[:,0]
        A = np.array([[5.0, 0.0],
                      [1.0, 9.0]])
        opponent = np.array([1.0, 0.0])   # always col 0
        # Expected payoffs: row0 = 5, row1 = 1 → best is row 0
        assert best_response(A, opponent) == 0

    def test_valid_index_range(self):
        A = np.random.rand(4, 3)
        opponent = np.array([0.2, 0.5, 0.3])
        result = best_response(A, opponent)
        assert 0 <= result < 4


# ─────────────────────────────────────────────────────────────────────────────
# sample_strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleStrategy:

    def test_returns_valid_index(self):
        sigma = np.array([0.3, 0.5, 0.2])
        for _ in range(50):
            assert sample_strategy(sigma) in [0, 1, 2]

    def test_pure_strategy_always_same(self):
        sigma = np.array([0.0, 1.0, 0.0])
        results = {sample_strategy(sigma) for _ in range(20)}
        assert results == {1}

    def test_handles_zero_vector_gracefully(self):
        sigma = np.array([0.0, 0.0, 0.0])
        result = sample_strategy(sigma)
        assert 0 <= result < 3

    def test_handles_negative_clipping(self):
        # Tiny negative due to floating point drift
        sigma = np.array([-1e-15, 0.5, 0.5])
        result = sample_strategy(sigma)
        assert 0 <= result < 3

    def test_returns_int(self):
        sigma = np.array([0.4, 0.6])
        assert isinstance(sample_strategy(sigma), int)


# ─────────────────────────────────────────────────────────────────────────────
# run_simulation
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_BASE_COLS = {
    "round", "defender_strategy", "attacker_strategy",
    "defender_payoff", "attacker_payoff",
    "cum_avg_defender", "cum_avg_attacker",
}


class TestRunSimulation:

    def test_returns_dataframe(self, pd_matrices, uniform_equilibria):
        A, B = pd_matrices
        df = run_simulation(A, B, uniform_equilibria, rounds=10)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, pd_matrices, uniform_equilibria):
        A, B = pd_matrices
        df = run_simulation(A, B, uniform_equilibria, rounds=25)
        assert len(df) == 25

    def test_required_columns_present(self, pd_matrices, uniform_equilibria):
        A, B = pd_matrices
        df = run_simulation(A, B, uniform_equilibria, rounds=5)
        assert EXPECTED_BASE_COLS.issubset(df.columns)

    def test_round_column_sequential(self, pd_matrices, uniform_equilibria):
        A, B = pd_matrices
        df = run_simulation(A, B, uniform_equilibria, rounds=10)
        assert list(df["round"]) == list(range(1, 11))

    def test_strategy_indices_in_valid_range(self, pd_matrices, uniform_equilibria):
        A, B = pd_matrices
        df = run_simulation(A, B, uniform_equilibria, rounds=50)
        assert df["defender_strategy"].between(0, 1).all()
        assert df["attacker_strategy"].between(0, 1).all()

    def test_payoffs_come_from_matrix(self, pd_matrices, pure_equilibria):
        A, B = pd_matrices
        # Pure NE (def=1, atk=1) → payoffs must be A[1,1]=1.0, B[1,1]=1.0
        df = run_simulation(A, B, pure_equilibria, rounds=10)
        assert (df["defender_payoff"] == 1.0).all()
        assert (df["attacker_payoff"] == 1.0).all()

    def test_cumulative_average_converges_pure_ne(self, pd_matrices, pure_equilibria):
        A, B = pd_matrices
        df = run_simulation(A, B, pure_equilibria, rounds=20)
        # With pure (1,1) strategy, cum_avg should be exactly 1.0 every round
        np.testing.assert_allclose(df["cum_avg_defender"].values, 1.0, atol=1e-9)
        np.testing.assert_allclose(df["cum_avg_attacker"].values, 1.0, atol=1e-9)

    def test_works_with_no_equilibria(self, pd_matrices):
        A, B = pd_matrices
        # Empty equilibria → falls back to uniform
        df = run_simulation(A, B, [], rounds=10)
        assert len(df) == 10

    def test_probability_columns_present(self, pd_matrices, uniform_equilibria):
        A, B = pd_matrices
        df = run_simulation(A, B, uniform_equilibria, rounds=5)
        assert "def_prob_0" in df.columns
        assert "def_prob_1" in df.columns
        assert "atk_prob_0" in df.columns
        assert "atk_prob_1" in df.columns

    def test_probability_columns_sum_to_one(self, pd_matrices, uniform_equilibria):
        A, B = pd_matrices
        df = run_simulation(A, B, uniform_equilibria, rounds=10)
        prob_sum = df["def_prob_0"] + df["def_prob_1"]
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-9)

    def test_3x3_game(self):
        A = np.array([[3.0, 1.0, 2.0],
                      [0.0, 4.0, 1.0],
                      [2.0, 0.0, 3.0]])
        B = -A  # zero-sum
        sigma = np.ones(3) / 3.0
        eq = [(sigma, sigma)]
        df = run_simulation(A, B, eq, rounds=15)
        assert len(df) == 15
        assert df["defender_strategy"].between(0, 2).all()


# ─────────────────────────────────────────────────────────────────────────────
# run_best_response_dynamics
# ─────────────────────────────────────────────────────────────────────────────

class TestRunBestResponseDynamics:

    def test_returns_dataframe(self, pd_matrices):
        A, B = pd_matrices
        df = run_best_response_dynamics(A, B, rounds=20)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, pd_matrices):
        A, B = pd_matrices
        df = run_best_response_dynamics(A, B, rounds=30)
        assert len(df) == 30

    def test_required_columns_present(self, pd_matrices):
        A, B = pd_matrices
        df = run_best_response_dynamics(A, B, rounds=10)
        assert EXPECTED_BASE_COLS.issubset(df.columns)

    def test_round_column_sequential(self, pd_matrices):
        A, B = pd_matrices
        df = run_best_response_dynamics(A, B, rounds=5)
        assert list(df["round"]) == [1, 2, 3, 4, 5]

    def test_strategy_indices_in_valid_range(self, pd_matrices):
        A, B = pd_matrices
        df = run_best_response_dynamics(A, B, rounds=20)
        assert df["defender_strategy"].between(0, 1).all()
        assert df["attacker_strategy"].between(0, 1).all()

    def test_payoffs_valid_matrix_entries(self, pd_matrices):
        A, B = pd_matrices
        valid_def = {A[i, j] for i in range(2) for j in range(2)}
        valid_atk = {B[i, j] for i in range(2) for j in range(2)}
        df = run_best_response_dynamics(A, B, rounds=10)
        assert all(p in valid_def for p in df["defender_payoff"])
        assert all(p in valid_atk for p in df["attacker_payoff"])

    def test_probability_columns_sum_to_one(self, pd_matrices):
        A, B = pd_matrices
        df = run_best_response_dynamics(A, B, rounds=10)
        prob_sum = df["def_prob_0"] + df["def_prob_1"]
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-9)
