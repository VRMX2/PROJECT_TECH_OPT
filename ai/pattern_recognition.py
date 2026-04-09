"""
ai/pattern_recognition.py
──────────────────────────
Attack pattern recognition module using scikit-learn.

Trains a RandomForestClassifier on historical attacker strategy sequences
to predict the attacker's next move. The model treats the problem as a
sequence classification task:

  Input features:  Last K attacker strategy indices (sliding window)
  Target:          Next attacker strategy index

Provides:
  • AttackPatternClassifier class
      .fit(history)     → trains on a list of strategy indices
      .predict(window)  → predicted next attacker strategy index
      .predict_proba(window) → probability distribution over strategies
      .is_trained       → bool
  • generate_synthetic_history(n_strategies, n_rounds) → list of int
      (used to warm-start the classifier in the UI)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Optional


class AttackPatternClassifier:
    """
    Predicts the next attacker strategy by learning from a history of past moves.

    Uses a sliding window of the last `window_size` moves as features.
    """

    def __init__(self, n_strategies: int, window_size: int = 3):
        """
        Args:
            n_strategies: Total number of attacker strategies.
            window_size:  Number of recent moves used as input features.
        """
        self.n_strategies  = n_strategies
        self.window_size   = window_size
        self.model         = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=5,
        )
        self.is_trained    = False
        self._classes      = list(range(n_strategies))

    # ─────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, history: list) -> bool:
        """
        Train the classifier on a sequence of attacker strategy indices.

        Args:
            history: List of int (strategy indices, 0-indexed).

        Returns:
            True if training succeeded, False if not enough data.
        """
        if len(history) < self.window_size + 2:
            self.is_trained = False
            return False

        X, y = self._extract_features(history)
        if len(X) < 2:
            self.is_trained = False
            return False

        # Ensure all classes are represented (add dummy rows if needed)
        X_aug = list(X)
        y_aug = list(y)
        for cls in self._classes:
            if cls not in y_aug:
                X_aug.append([0] * self.window_size)
                y_aug.append(cls)

        self.model.fit(X_aug, y_aug)
        self.is_trained = True
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, window: list) -> int:
        """
        Predict the next attacker strategy.

        Args:
            window: List of the last `window_size` attacker strategy indices.

        Returns:
            Predicted next strategy index, or random fallback if not trained.
        """
        if not self.is_trained:
            return np.random.randint(0, self.n_strategies)
        x = self._pad_window(window)
        return int(self.model.predict([x])[0])

    def predict_proba(self, window: list) -> np.ndarray:
        """
        Return probability distribution over all strategies.

        Args:
            window: List of recent attacker strategy indices.

        Returns:
            np.ndarray of shape (n_strategies,) summing to 1.
        """
        if not self.is_trained:
            return np.ones(self.n_strategies) / self.n_strategies
        x = self._pad_window(window)
        proba = self.model.predict_proba([x])[0]
        # Align with class ordering
        result = np.zeros(self.n_strategies)
        for idx, cls in enumerate(self.model.classes_):
            result[int(cls)] = proba[idx]
        return result

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importances (one per window position)."""
        if not self.is_trained:
            return np.ones(self.window_size) / self.window_size
        return self.model.feature_importances_

    # ─────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────

    def _extract_features(self, history: list):
        """Build sliding-window feature matrix from history."""
        X, y = [], []
        for i in range(len(history) - self.window_size):
            window = history[i : i + self.window_size]
            target = history[i + self.window_size]
            X.append(window)
            y.append(target)
        return np.array(X), np.array(y)

    def _pad_window(self, window: list) -> list:
        """Ensure the window is exactly window_size elements long."""
        window = list(window)
        if len(window) >= self.window_size:
            return window[-self.window_size:]
        # Pad left with zeros
        return [0] * (self.window_size - len(window)) + window

    def reset(self, n_strategies: Optional[int] = None):
        """Reset the classifier."""
        if n_strategies:
            self.n_strategies = n_strategies
            self._classes = list(range(n_strategies))
        self.model = RandomForestClassifier(
            n_estimators=50, random_state=42, max_depth=5
        )
        self.is_trained = False


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic history generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_history(n_strategies: int, n_rounds: int = 200) -> list:
    """
    Generate a synthetic attack history with realistic patterns:
      - Attacker tends to repeat successful strategies
      - Occasional random exploration

    Args:
        n_strategies: Number of available attacker strategies.
        n_rounds:     Number of rounds to simulate.

    Returns:
        List of strategy indices.
    """
    history = []
    current = np.random.randint(0, n_strategies)

    for _ in range(n_rounds):
        history.append(current)
        # 70% chance to stick or shift to adjacent strategy, 30% random
        if np.random.random() < 0.7:
            shift = np.random.choice([-1, 0, 1])
            current = int(np.clip(current + shift, 0, n_strategies - 1))
        else:
            current = np.random.randint(0, n_strategies)

    return history
