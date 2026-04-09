"""
ai/q_learning.py
────────────────
Tabular Q-learning defender agent.

The Q-learning agent represents the DEFENDER, learning over repeated game rounds
to minimise attacker success (maximise its own payoffs) by observing attacker moves.

State  : Last attacker strategy index (observable signal).
Action : Defender strategy index to play.
Reward : Defender's payoff from the game matrix A[action, atk_action].

Provides:
  • QLearningAgent class
      .select_action(state)                → int
      .update(state, action, reward, next_state)
      .get_q_table()                       → np.ndarray
      .reset()
"""

import numpy as np
import random


class QLearningAgent:
    """
    Tabular epsilon-greedy Q-learning agent for the Defender.

    Q-table dimensions: (n_states, n_actions)
      where n_states = number of attacker strategies (observable states)
      and   n_actions = number of defender strategies.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,       # learning rate
        gamma: float = 0.9,       # discount factor
        epsilon: float = 1.0,     # initial exploration rate
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        """
        Args:
            n_states:       Number of distinct attacker strategies (states).
            n_actions:      Number of distinct defender strategies (actions).
            alpha:          Learning rate (step size).
            gamma:          Discount factor for future rewards.
            epsilon:        Initial exploration probability (greedy epsilon).
            epsilon_decay:  Multiplicative decay per step.
            epsilon_min:    Minimum exploration probability.
        """
        self.n_states   = n_states
        self.n_actions  = n_actions
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Q-table initialised to zeros
        self.Q = np.zeros((n_states, n_actions), dtype=float)

        # Training history for visualisation
        self.reward_history: list = []
        self.epsilon_history: list = []

    # ─────────────────────────────────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────────────────────────────────

    def select_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Current state (last attacker strategy index).

        Returns:
            Defender strategy index to play.
        """
        state = int(state) % self.n_states  # guard bounds

        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: best Q-value action
            return int(np.argmax(self.Q[state]))

    # ─────────────────────────────────────────────────────────────────────
    # Q-table update
    # ─────────────────────────────────────────────────────────────────────

    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Apply a Q-learning update step (Bellman equation).

        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

        Args:
            state:      Previous state.
            action:     Action taken (defender strategy index).
            reward:     Received reward (defender's payoff).
            next_state: Next state (current attacker strategy index).
        """
        state      = int(state)      % self.n_states
        next_state = int(next_state) % self.n_states
        action     = int(action)     % self.n_actions

        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Record for visualisation
        self.reward_history.append(float(reward))
        self.epsilon_history.append(self.epsilon)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def get_q_table(self) -> np.ndarray:
        """Return the current Q-table."""
        return self.Q.copy()

    def get_policy(self) -> np.ndarray:
        """Return the greedy policy (best action per state)."""
        return np.argmax(self.Q, axis=1)

    def get_mixed_strategy(self, state: int) -> np.ndarray:
        """
        Convert Q-values for a given state to a softmax mixed strategy.
        Useful for blending exploration into the reported strategy distribution.
        """
        state = int(state) % self.n_states
        q_vals = self.Q[state]
        # Softmax with temperature 1
        exp_q = np.exp(q_vals - np.max(q_vals))
        return exp_q / exp_q.sum()

    def reset(self):
        """Reset the Q-table and training history."""
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=float)
        self.epsilon = 1.0
        self.reward_history = []
        self.epsilon_history = []
