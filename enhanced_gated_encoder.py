"""
Enhanced Surprise-Gated Encoder with temporal detection and entity tracking.

Extends the pure surprise-gating approach with two bypass mechanisms:

  1. Temporal bypass — messages containing date patterns, time expressions,
     or temporal keywords are stored unconditionally. These are critical for
     answering "when" questions, which pure surprise-gating devastates because
     routine timestamped updates look "unsurprising" to cosine distance.

  2. Entity novelty bypass — messages introducing entities (people, places,
     organizations) that a speaker hasn't mentioned before are stored
     unconditionally. Novel entities signal new information even when the
     embedding vector isn't far from existing memories.

The storage decision becomes:
    store = (surprise > threshold) OR has_temporal_markers OR has_novel_entities

Every stored memory is a MemoryRecord carrying metadata for future use:
retrieval_weight, inhibition_weight, etc. Retrieval uses weighted scoring:
    final_score = cosine_similarity * retrieval_weight * (1 - inhibition_weight)
Currently weights are initialized to neutral values (1.0 / 0.0), so this is
pure plumbing — but it's ready for decay, interference, and consolidation.
"""

import re
import numpy as np
import spacy
from dataclasses import dataclass, field
from typing import Optional


# ─── MemoryRecord ────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """A single stored memory with full metadata."""
    text: str
    embedding: np.ndarray
    retrieval_weight: float = 1.0
    retrieval_count: int = 0
    last_retrieved: Optional[float] = None
    associations: list[int] = field(default_factory=list)
    inhibited_by: Optional[int] = None
    inhibition_weight: float = 0.0
    created_at: str = ""
    surprise_score: float = 0.0
    temporal_salience: bool = False
    entity_novelty: bool = False


# ─── Temporal Detector ───────────────────────────────────────────────────────

_MONTH = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
_DOW = r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'

TEMPORAL_PATTERNS = [
    # "May 7, 2023", "January 15th", "December 2021"
    re.compile(rf'\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b', re.I),
    # "7 May 2023"
    re.compile(rf'\b\d{{1,2}}\s+{_MONTH}(?:\s+\d{{4}})?\b', re.I),
    # "5/7/2023", "07-05-23"
    re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
    # "2023-05-07"
    re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
    # "last Monday", "next week", "this year", "past month"
    re.compile(rf'\b(?:last|next|this|past)\s+(?:{_DOW}|week|month|year|spring|summer|fall|autumn|winter|semester|quarter)\b', re.I),
    # "yesterday", "tomorrow", "today", "tonight"
    re.compile(r'\b(?:yesterday|tomorrow|today|tonight)\b', re.I),
    # "three months ago", "2 weeks from now", "a year later"
    re.compile(r'\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:day|week|month|year|hour|minute|decade)s?\s+(?:ago|from\s+now|later|earlier|back)\b', re.I),
    # "at 3pm", "at 11:30 AM"
    re.compile(r'\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b', re.I),
    # "in the morning", "in the evening"
    re.compile(r'\bin\s+the\s+(?:morning|afternoon|evening|night)\b', re.I),
    # "at noon", "at midnight"
    re.compile(r'\bat\s+(?:noon|midnight|dawn|dusk)\b', re.I),
    # Temporal keywords (user-specified list)
    re.compile(r'\b(?:started|began|moved|changed|recently|just|ago|since)\b', re.I),
]


class TemporalDetector:
    """Lightweight regex-based detector for temporal expressions."""

    def detect(self, text: str) -> bool:
        for pattern in TEMPORAL_PATTERNS:
            if pattern.search(text):
                return True
        return False


# ─── Entity Tracker ──────────────────────────────────────────────────────────

TRACKED_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART"}


class EntityTracker:
    """
    Tracks seen entities per speaker using spaCy NER.
    Returns whether a message introduces any entity the speaker hasn't
    mentioned before.
    """

    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        self._seen: dict[str, set[str]] = {}

    def reset(self, speaker: str | None = None):
        if speaker:
            self._seen.pop(speaker, None)
        else:
            self._seen.clear()

    def check(self, text: str, speaker: str) -> tuple[bool, list[str]]:
        """
        Returns (has_novel_entities, list_of_novel_entity_strings).
        """
        if speaker not in self._seen:
            self._seen[speaker] = set()

        doc = self._nlp(text)
        entities = {
            (ent.text.strip().lower(), ent.label_)
            for ent in doc.ents
            if ent.label_ in TRACKED_ENTITY_TYPES and len(ent.text.strip()) > 1
        }

        novel = []
        for ent_text, ent_label in entities:
            key = f"{ent_text}|{ent_label}"
            if key not in self._seen[speaker]:
                self._seen[speaker].add(key)
                novel.append(f"{ent_text} ({ent_label})")

        return len(novel) > 0, novel


# ─── Enhanced Gate Stats ─────────────────────────────────────────────────────

@dataclass
class EnhancedGateStats:
    """Tracks gating decisions including bypass reasons."""
    total_seen: int = 0
    total_stored: int = 0
    stored_by_surprise: int = 0
    stored_by_temporal: int = 0
    stored_by_entity: int = 0
    stored_by_warmup: int = 0
    surprise_scores: list = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if self.total_seen == 0:
            return 0.0
        return 1.0 - (self.total_stored / self.total_seen)

    @property
    def mean_surprise(self) -> float:
        return float(np.mean(self.surprise_scores)) if self.surprise_scores else 0.0

    def summary(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "total_stored": self.total_stored,
            "compression_ratio": self.compression_ratio,
            "mean_surprise": self.mean_surprise,
            "stored_by_surprise": self.stored_by_surprise,
            "stored_by_temporal": self.stored_by_temporal,
            "stored_by_entity": self.stored_by_entity,
            "stored_by_warmup": self.stored_by_warmup,
        }


# ─── Enhanced Gated Encoder ─────────────────────────────────────────────────

class EnhancedGatedEncoder:
    """
    Gating with three storage pathways:

      1. Surprise gate — cosine novelty (same as SurpriseGatedEncoder)
      2. Temporal bypass — regex detects temporal content
      3. Entity novelty bypass — spaCy NER detects new entities

    Parameters are the same as SurpriseGatedEncoder (threshold, mode, metric,
    warmup) — the temporal and entity bypasses have no tunable parameters.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        mode: str = "fixed",
        metric: str = "nearest_neighbor",
        warmup: int = 3,
        temporal_detector: TemporalDetector | None = None,
        entity_tracker: EntityTracker | None = None,
    ):
        self.threshold = threshold
        self.mode = mode
        self.metric = metric
        self.warmup = warmup

        self.temporal_detector = temporal_detector or TemporalDetector()
        self.entity_tracker = entity_tracker

        self._stored_embeddings: list[np.ndarray] = []
        self._centroid: np.ndarray | None = None
        self.stats = EnhancedGateStats()

    def reset(self):
        self._stored_embeddings = []
        self._centroid = None
        self.stats = EnhancedGateStats()

    # ── surprise computation (same math as SurpriseGatedEncoder) ──

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _compute_surprise(self, embedding: np.ndarray) -> float:
        if not self._stored_embeddings:
            return 1.0

        if self.metric == "nearest_neighbor":
            sims = [self._cosine_sim(embedding, s) for s in self._stored_embeddings]
            return 1.0 - max(sims)

        if self.metric == "centroid":
            if self._centroid is None:
                return 1.0
            centroid_norm = self._centroid / (np.linalg.norm(self._centroid) + 1e-10)
            return 1.0 - self._cosine_sim(embedding, centroid_norm)

        raise ValueError(f"Unknown metric: {self.metric}")

    def _get_effective_threshold(self) -> float:
        if self.mode == "fixed":
            return self.threshold
        if self.mode == "adaptive":
            if len(self.stats.surprise_scores) < self.warmup:
                return 0.0
            return float(np.percentile(self.stats.surprise_scores, self.threshold * 100))
        raise ValueError(f"Unknown mode: {self.mode}")

    def _update_centroid(self, embedding: np.ndarray):
        n = len(self._stored_embeddings)
        if self._centroid is None:
            self._centroid = embedding.copy()
        else:
            self._centroid = (self._centroid * (n - 1) + embedding) / n

    # ── main gate ──

    def gate(
        self,
        embedding: np.ndarray,
        text: str,
        speaker: str,
    ) -> tuple[bool, float, bool, bool]:
        """
        Decide whether to store this message.

        Returns (should_store, surprise_score, temporal_salience, entity_novelty).
        """
        self.stats.total_seen += 1

        # Always store during warmup
        if len(self._stored_embeddings) < self.warmup:
            surprise = self._compute_surprise(embedding)
            self.stats.surprise_scores.append(surprise)
            self._stored_embeddings.append(embedding)
            self._update_centroid(embedding)
            self.stats.total_stored += 1
            self.stats.stored_by_warmup += 1
            return True, surprise, False, False

        surprise = self._compute_surprise(embedding)
        self.stats.surprise_scores.append(surprise)

        # Three pathways
        surprise_pass = surprise >= self._get_effective_threshold()
        temporal_pass = self.temporal_detector.detect(text)
        entity_pass = False
        if self.entity_tracker is not None:
            entity_pass, _ = self.entity_tracker.check(text, speaker)

        should_store = surprise_pass or temporal_pass or entity_pass

        if should_store:
            self._stored_embeddings.append(embedding)
            self._update_centroid(embedding)
            self.stats.total_stored += 1
            if surprise_pass:
                self.stats.stored_by_surprise += 1
            if temporal_pass:
                self.stats.stored_by_temporal += 1
            if entity_pass:
                self.stats.stored_by_entity += 1

        return should_store, surprise, temporal_pass, entity_pass
