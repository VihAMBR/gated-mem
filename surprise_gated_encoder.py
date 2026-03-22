"""
Surprise-Gated Encoder for memory systems.

Core idea: human memory doesn't store everything — it stores what's *surprising*.
If someone tells you the same thing for the 5th time, you don't form a new memory.
But if they mention something genuinely new, that sticks.

This module implements that intuition:
  1. For each incoming message, embed it
  2. Compare against existing stored memory embeddings
  3. Compute a "surprise score" — how novel is this vs what we already know?
  4. If surprise > threshold → encode (store it)
  5. If surprise < threshold → skip (redundant)

Two surprise metrics:
  - nearest_neighbor: cosine distance to closest existing memory
  - centroid: cosine distance to the mean of all existing memory embeddings

Two threshold strategies:
  - fixed: a static float (e.g. 0.3)
  - adaptive: percentile-based on the surprise scores seen so far,
    so the gate calibrates itself to the conversation's natural variability
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class GateStats:
    """Tracks what the gate did — essential for analysis."""
    total_seen: int = 0
    total_stored: int = 0
    surprise_scores: list = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if self.total_seen == 0:
            return 0.0
        return 1.0 - (self.total_stored / self.total_seen)

    @property
    def mean_surprise(self) -> float:
        return float(np.mean(self.surprise_scores)) if self.surprise_scores else 0.0


class SurpriseGatedEncoder:
    """
    Decides whether an incoming message is worth storing based on how
    "surprising" it is relative to what's already in memory.

    Parameters
    ----------
    threshold : float
        Surprise score above which a message gets stored.
        For fixed mode: used directly (0.0 = store everything, 1.0 = store nothing).
        For adaptive mode: used as the percentile (e.g. 0.5 = median).
    mode : str
        "fixed" or "adaptive"
    metric : str
        "nearest_neighbor" — surprise = 1 - max_cosine_sim to any stored memory
        "centroid" — surprise = 1 - cosine_sim to mean embedding
    warmup : int
        Always store the first N messages unconditionally (need some memories
        before the gate can make meaningful comparisons).
    """

    def __init__(
        self,
        threshold: float = 0.3,
        mode: str = "fixed",
        metric: str = "nearest_neighbor",
        warmup: int = 3,
    ):
        self.threshold = threshold
        self.mode = mode
        self.metric = metric
        self.warmup = warmup
        self.reset()

    def reset(self):
        """Clear state for a new conversation / speaker."""
        self.stored_embeddings: list[np.ndarray] = []
        self.centroid: np.ndarray | None = None
        self.stats = GateStats()

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors is just dot product."""
        return float(np.dot(a, b))

    def _compute_surprise(self, embedding: np.ndarray) -> float:
        """
        How surprising is this embedding relative to what's stored?
        Returns a float in [0, 1] where 1 = maximally surprising (nothing similar).
        """
        if not self.stored_embeddings:
            return 1.0

        if self.metric == "nearest_neighbor":
            sims = [self._cosine_sim(embedding, stored) for stored in self.stored_embeddings]
            return 1.0 - max(sims)

        elif self.metric == "centroid":
            if self.centroid is None:
                return 1.0
            centroid_norm = self.centroid / (np.linalg.norm(self.centroid) + 1e-10)
            return 1.0 - self._cosine_sim(embedding, centroid_norm)

        raise ValueError(f"Unknown metric: {self.metric}")

    def _get_effective_threshold(self) -> float:
        """
        For fixed mode, return the threshold directly.
        For adaptive mode, compute the threshold as a percentile of recent
        surprise scores — so the gate adapts to the conversation's variability.
        """
        if self.mode == "fixed":
            return self.threshold

        if self.mode == "adaptive":
            if len(self.stats.surprise_scores) < self.warmup:
                return 0.0  # during warmup, store everything
            percentile = self.threshold * 100  # e.g. 0.5 → 50th percentile
            return float(np.percentile(self.stats.surprise_scores, percentile))

        raise ValueError(f"Unknown mode: {self.mode}")

    def _update_centroid(self, embedding: np.ndarray):
        """Incrementally update the centroid (running mean)."""
        n = len(self.stored_embeddings)
        if self.centroid is None:
            self.centroid = embedding.copy()
        else:
            self.centroid = (self.centroid * (n - 1) + embedding) / n

    def gate(self, embedding: np.ndarray) -> tuple[bool, float]:
        """
        Decide whether to store this embedding.

        Returns (should_store, surprise_score).
        """
        self.stats.total_seen += 1

        # Warmup: always store the first N
        if len(self.stored_embeddings) < self.warmup:
            surprise = self._compute_surprise(embedding)
            self.stats.surprise_scores.append(surprise)
            self.stored_embeddings.append(embedding)
            self._update_centroid(embedding)
            self.stats.total_stored += 1
            return True, surprise

        surprise = self._compute_surprise(embedding)
        self.stats.surprise_scores.append(surprise)

        effective_threshold = self._get_effective_threshold()

        if surprise >= effective_threshold:
            self.stored_embeddings.append(embedding)
            self._update_centroid(embedding)
            self.stats.total_stored += 1
            return True, surprise

        return False, surprise
