"""Early stopping monitor for training loops."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when monitored metric stops improving.

    Args:
        patience: epochs to wait after last improvement
        min_delta: minimum change to qualify as improvement
        mode: "min" (lower=better, e.g. val_loss) or "max" (higher=better, e.g. FID-inverse)
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def step(self, metric: float) -> bool:
        """Update with new metric value. Returns True if training should stop."""
        if self._is_improvement(metric):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered: no improvement for %d epochs (best=%.6f)",
                    self.patience,
                    self.best_score,
                )

        return self.should_stop

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.should_stop = False
