"""
Neuroplastic Memory System — four biologically-inspired plasticity mechanisms.

Mechanism 1: Retrieval Strengthening & Decay (LTP / LTD)
    Memories that prove useful during retrieval get weight boosts.
    All memories decay gently over time. The index self-organizes so
    frequently useful memories float to the top and noise sinks.

Mechanism 2: Associative Linking (Hebbian / Structural Plasticity)
    Memories that are co-retrieved together become linked. During retrieval,
    a one-hop expansion surfaces associated memories that FAISS alone
    wouldn't return — critical for multi-hop questions.

Mechanism 3: Belief Revision through Inhibition (Functional Reorganization)
    When a newer memory contradicts an older one (high embedding similarity,
    different session), the older one gets inhibited. Queries about past
    states can temporarily lift the inhibition.

Mechanism 4: Memory Consolidation (Sleep-time Reorganization)
    Periodic offline pass: merge near-duplicates, decay unretrieved memories
    harder, and generate centroid-based abstract "summary" memories from
    clusters of related memories.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import numpy as np
import faiss

from enhanced_gated_encoder import MemoryRecord


# ─── Mechanism 1: Retrieval Strengthening & Decay ────────────────────────────

class RetrievalPlasticity:
    """LTP (strengthen on correct retrieval) + LTD (global decay)."""

    def __init__(self, boost: float = 0.05, decay_rate: float = 0.01,
                 weight_floor: float = 0.1, weight_ceiling: float = 5.0):
        self.boost = boost
        self.decay_rate = decay_rate
        self.weight_floor = weight_floor
        self.weight_ceiling = weight_ceiling
        self.total_strengthened = 0
        self.total_decays_applied = 0

    def strengthen(self, records: list[MemoryRecord], retrieved_indices: list[int]):
        """Boost retrieved memories that contributed to a correct answer."""
        for idx in retrieved_indices:
            if idx < len(records):
                rec = records[idx]
                rec.retrieval_weight = min(
                    rec.retrieval_weight * (1 + self.boost),
                    self.weight_ceiling,
                )
                rec.retrieval_count += 1
                rec.last_retrieved = time.time()
                self.total_strengthened += 1

    def decay_all(self, records: list[MemoryRecord]):
        """Apply gentle decay to all memories. Recently strengthened ones barely feel it."""
        for rec in records:
            rec.retrieval_weight *= (1 - self.decay_rate)
            rec.retrieval_weight = max(rec.retrieval_weight, self.weight_floor)
        self.total_decays_applied += 1

    def get_weight_distribution(self, records: list[MemoryRecord]) -> dict:
        if not records:
            return {}
        weights = [r.retrieval_weight for r in records]
        return {
            "min": float(min(weights)),
            "max": float(max(weights)),
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "median": float(np.median(weights)),
            "above_1": sum(1 for w in weights if w > 1.05),
            "below_1": sum(1 for w in weights if w < 0.95),
        }


# ─── Mechanism 2: Associative Linking ────────────────────────────────────────

class AssociationGraph:
    """Hebbian co-retrieval tracking with one-hop expansion during retrieval."""

    def __init__(self, link_threshold: int = 3, max_expansion: int = 10,
                 bonus_per_strength: float = 0.05):
        self.co_retrieval: dict[tuple[int, int], int] = {}
        self.link_threshold = link_threshold
        self.max_expansion = max_expansion
        self.bonus_per_strength = bonus_per_strength
        self.total_links_formed = 0
        self.total_expansions = 0

    def record_co_retrieval(self, retrieved_indices: list[int]):
        """Increment co-retrieval count for all pairs in the retrieved set."""
        for i, a in enumerate(retrieved_indices):
            for b in retrieved_indices[i + 1:]:
                pair = (min(a, b), max(a, b))
                self.co_retrieval[pair] = self.co_retrieval.get(pair, 0) + 1

    def get_associations(self, mem_idx: int) -> list[tuple[int, int]]:
        """Return (associated_idx, strength) for all strong links to mem_idx."""
        associations = []
        for (a, b), count in self.co_retrieval.items():
            if count >= self.link_threshold:
                if a == mem_idx:
                    associations.append((b, count))
                elif b == mem_idx:
                    associations.append((a, count))
        return associations

    def expand_retrieval(
        self,
        candidates: list[tuple[float, int]],
        records: list[MemoryRecord],
        top_k: int,
    ) -> list[tuple[float, int]]:
        """
        One-hop association expansion: for top candidates, pull in linked
        memories that weren't in the initial set.
        """
        retrieved_set = {idx for _, idx in candidates}
        bonus_entries = []

        expand_count = min(self.max_expansion, len(candidates))
        for _, parent_idx in candidates[:expand_count]:
            for assoc_idx, strength in self.get_associations(parent_idx):
                if assoc_idx not in retrieved_set and assoc_idx < len(records):
                    bonus_score = self.bonus_per_strength * strength
                    bonus_entries.append((bonus_score, assoc_idx))
                    retrieved_set.add(assoc_idx)
                    self.total_expansions += 1

        all_candidates = list(candidates) + bonus_entries
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        return all_candidates[:top_k]

    @property
    def num_active_links(self) -> int:
        return sum(1 for count in self.co_retrieval.values()
                   if count >= self.link_threshold)

    def prune(self, min_strength: int = 1):
        """Remove weak associations that were never reinforced."""
        self.co_retrieval = {
            pair: count for pair, count in self.co_retrieval.items()
            if count >= min_strength
        }


# ─── Mechanism 3: Belief Revision through Inhibition ────────────────────────

PAST_INDICATORS = [
    "used to", "originally", "before", "previously", "at first",
    "initially", "back when", "former", "old", "earlier",
]


class BeliefRevisionDetector:
    """
    Detect when newer memories supersede older ones and apply inhibition.
    Uses embedding similarity + session timestamp heuristic.
    """

    def __init__(self, similarity_threshold: float = 0.85,
                 inhibition_strength: float = 0.7,
                 inhibition_cap: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.inhibition_strength = inhibition_strength
        self.inhibition_cap = inhibition_cap
        self.total_inhibitions = 0

    def detect_and_inhibit(self, records: list[MemoryRecord]):
        """
        For each record, use FAISS to find its nearest neighbors, then check
        if any high-similarity neighbor from a different session should inhibit it.
        Much faster than O(n^2) brute-force for large memory banks.
        """
        if len(records) < 2:
            return

        embs = np.array([r.embedding for r in records], dtype=np.float32)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        k = min(10, len(records))
        scores, indices = index.search(embs, k)

        for i in range(len(records)):
            if records[i].inhibition_weight >= self.inhibition_cap:
                continue
            for rank in range(1, k):
                j = int(indices[i][rank])
                sim = float(scores[i][rank])
                if sim < self.similarity_threshold:
                    break
                if j > i and records[i].created_at != records[j].created_at:
                    records[i].inhibition_weight = min(
                        records[i].inhibition_weight + self.inhibition_strength,
                        self.inhibition_cap,
                    )
                    records[i].inhibited_by = j
                    self.total_inhibitions += 1
                    break

    @staticmethod
    def is_asking_about_past(query: str) -> bool:
        q_lower = query.lower()
        return any(ind in q_lower for ind in PAST_INDICATORS)

    @staticmethod
    def effective_inhibition(record: MemoryRecord, asking_past: bool) -> float:
        """Temporarily reduce inhibition for queries about past states."""
        if asking_past and record.inhibited_by is not None:
            return record.inhibition_weight * 0.3
        return record.inhibition_weight


# ─── Mechanism 4: Memory Consolidation ───────────────────────────────────────

class ConsolidationEngine:
    """
    Periodic offline reorganization:
      1. Merge near-duplicate memories (keep highest weight, transfer metadata)
      2. Heavier decay on unretrieved memories
      3. Generate centroid-based abstract summaries from memory clusters
    """

    def __init__(self, merge_threshold: float = 0.92,
                 cluster_threshold: float = 0.88,
                 min_cluster_size: int = 3,
                 abstraction_weight: float = 1.5):
        self.merge_threshold = merge_threshold
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.abstraction_weight = abstraction_weight
        self.merges_performed = 0
        self.abstractions_created = 0

    def consolidate(
        self, records: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Run full consolidation pass. Returns updated list with abstractions appended."""
        self._merge_duplicates(records)
        self._extra_decay(records)
        abstractions = self._generate_abstractions(records)
        return records + abstractions

    def _merge_duplicates(self, records: list[MemoryRecord]):
        """
        Use FAISS to find near-duplicate pairs (cosine > merge_threshold),
        keep the one with higher retrieval_weight, inhibit the weaker one.
        """
        if len(records) < 2:
            return

        embs = np.array([r.embedding for r in records], dtype=np.float32)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        k = min(5, len(records))
        scores, indices = index.search(embs, k)

        inhibited = set()
        for i in range(len(records)):
            if i in inhibited:
                continue
            for rank in range(1, k):
                j = int(indices[i][rank])
                sim = float(scores[i][rank])
                if sim < self.merge_threshold:
                    break
                if j in inhibited or j <= i:
                    continue
                if records[i].retrieval_weight >= records[j].retrieval_weight:
                    winner, loser = i, j
                else:
                    winner, loser = j, i
                records[winner].temporal_salience = (
                    records[winner].temporal_salience or records[loser].temporal_salience
                )
                records[winner].entity_novelty = (
                    records[winner].entity_novelty or records[loser].entity_novelty
                )
                records[loser].inhibition_weight = min(
                    records[loser].inhibition_weight + 0.7, 0.95
                )
                records[loser].inhibited_by = winner
                inhibited.add(loser)
                self.merges_performed += 1

    def _extra_decay(self, records: list[MemoryRecord]):
        """Stronger decay for memories that have never been retrieved."""
        for rec in records:
            if rec.retrieval_count == 0:
                rec.retrieval_weight *= 0.95
                rec.retrieval_weight = max(rec.retrieval_weight, 0.1)

    def _generate_abstractions(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """
        Use FAISS to find clusters of similar active memories. For each cluster
        with 3+ members, create a centroid-based abstract summary.
        """
        if len(records) < self.min_cluster_size:
            return []

        active = [(i, r) for i, r in enumerate(records) if r.inhibition_weight < 0.5]
        if len(active) < self.min_cluster_size:
            return []

        active_indices = [i for i, _ in active]
        embs = np.array([records[i].embedding for i in active_indices], dtype=np.float32)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        k = min(10, len(active))
        scores, nn_indices = index.search(embs, k)

        assigned = set()
        clusters = []

        for local_a in range(len(active)):
            global_a = active_indices[local_a]
            if global_a in assigned:
                continue
            cluster = [global_a]
            assigned.add(global_a)
            for rank in range(1, k):
                local_b = int(nn_indices[local_a][rank])
                sim = float(scores[local_a][rank])
                if sim < self.cluster_threshold:
                    break
                global_b = active_indices[local_b]
                if global_b not in assigned:
                    cluster.append(global_b)
                    assigned.add(global_b)
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        abstractions = []
        for cluster_indices in clusters:
            cluster_embs = np.array([records[i].embedding for i in cluster_indices])
            centroid = cluster_embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

            snippet = records[cluster_indices[0]].text[:60] + "..."
            abstract_text = (
                f"[Consolidated pattern from {len(cluster_indices)} memories] "
                f"e.g. {snippet}"
            )

            abstract = MemoryRecord(
                text=abstract_text,
                embedding=centroid.astype(np.float32),
                retrieval_weight=self.abstraction_weight,
                created_at=records[cluster_indices[-1]].created_at,
            )
            abstractions.append(abstract)
            self.abstractions_created += 1

        return abstractions


# ─── Neuroplastic Memory System (orchestrates all 4 mechanisms) ──────────────

@dataclass
class PlasticityConfig:
    """Feature flags and parameters for each mechanism."""
    enable_ltp: bool = True
    enable_associations: bool = True
    enable_inhibition: bool = True
    enable_consolidation: bool = True

    # LTP/LTD params
    ltp_boost: float = 0.05
    ltd_decay_rate: float = 0.01
    decay_every_n: int = 10

    # Association params
    assoc_link_threshold: int = 3
    assoc_max_expansion: int = 10
    assoc_bonus: float = 0.05

    # Inhibition params
    inhib_similarity: float = 0.85
    inhib_strength: float = 0.7
    inhib_cap: float = 0.95

    # Consolidation params
    consolidation_merge_threshold: float = 0.92
    consolidation_cluster_threshold: float = 0.88
    consolidation_min_cluster: int = 3
    consolidation_abstraction_weight: float = 1.5


class NeuroplasticMemory:
    """
    Orchestrates the four plasticity mechanisms over a set of MemoryRecords.

    Usage:
        npm = NeuroplasticMemory(config)
        npm.ingest(records)         # Phase 1: encoding + inhibition
        results = npm.retrieve(q, index, records, top_k)  # Phase 2
        npm.feedback(records, retrieved_indices, correct)  # Phase 3
        npm.consolidate(records)    # Phase 4 (periodic)
    """

    def __init__(self, config: PlasticityConfig | None = None):
        self.config = config or PlasticityConfig()
        self.questions_since_decay = 0
        self.questions_answered = 0

        self.ltp = RetrievalPlasticity(
            boost=self.config.ltp_boost,
            decay_rate=self.config.ltd_decay_rate,
        ) if self.config.enable_ltp else None

        self.associations = AssociationGraph(
            link_threshold=self.config.assoc_link_threshold,
            max_expansion=self.config.assoc_max_expansion,
            bonus_per_strength=self.config.assoc_bonus,
        ) if self.config.enable_associations else None

        self.inhibitor = BeliefRevisionDetector(
            similarity_threshold=self.config.inhib_similarity,
            inhibition_strength=self.config.inhib_strength,
            inhibition_cap=self.config.inhib_cap,
        ) if self.config.enable_inhibition else None

        self.consolidator = ConsolidationEngine(
            merge_threshold=self.config.consolidation_merge_threshold,
            cluster_threshold=self.config.consolidation_cluster_threshold,
            min_cluster_size=self.config.consolidation_min_cluster,
            abstraction_weight=self.config.consolidation_abstraction_weight,
        ) if self.config.enable_consolidation else None

    # ── Phase 1: Post-encoding inhibition ──

    def apply_inhibition(self, records: list[MemoryRecord]):
        if self.inhibitor:
            self.inhibitor.detect_and_inhibit(records)

    # ── Phase 2: Neuroplastic retrieval ──

    def score_candidates(
        self,
        query: str,
        raw_scores: np.ndarray,
        indices: np.ndarray,
        records: list[MemoryRecord],
        top_k: int,
    ) -> list[tuple[float, int]]:
        """
        Re-score FAISS candidates with plasticity weights.
        Returns sorted (score, index) pairs.
        """
        asking_past = (
            self.inhibitor is not None
            and BeliefRevisionDetector.is_asking_about_past(query)
        )

        candidates = []
        for score, idx in zip(raw_scores, indices):
            if idx >= len(records):
                continue
            rec = records[idx]

            weight = rec.retrieval_weight
            if self.inhibitor:
                eff_inhib = BeliefRevisionDetector.effective_inhibition(rec, asking_past)
            else:
                eff_inhib = rec.inhibition_weight

            final_score = float(score) * weight * (1.0 - eff_inhib)
            candidates.append((final_score, int(idx)))

        candidates.sort(key=lambda x: x[0], reverse=True)

        # Association expansion
        if self.associations:
            candidates = self.associations.expand_retrieval(candidates, records, top_k)

        return candidates[:top_k]

    # ── Phase 3: Feedback after judging ──

    def feedback(self, records: list[MemoryRecord],
                 retrieved_indices: list[int], correct: bool):
        """Called after each question is answered and judged."""
        self.questions_answered += 1
        self.questions_since_decay += 1

        if self.associations:
            self.associations.record_co_retrieval(retrieved_indices)

        if correct and self.ltp:
            self.ltp.strengthen(records, retrieved_indices)

        if self.ltp and self.questions_since_decay >= self.config.decay_every_n:
            self.ltp.decay_all(records)
            self.questions_since_decay = 0

    # ── Phase 4: Consolidation ──

    def consolidate(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Run consolidation and return new list (originals + abstractions)."""
        if self.consolidator:
            return self.consolidator.consolidate(records)
        return records

    # ── Stats ──

    def stats_summary(self, user_records: list[MemoryRecord],
                      asst_records: list[MemoryRecord]) -> dict:
        result = {
            "questions_answered": self.questions_answered,
        }
        if self.ltp:
            result["ltp"] = {
                "total_strengthened": self.ltp.total_strengthened,
                "decay_rounds": self.ltp.total_decays_applied,
                "user_weight_dist": self.ltp.get_weight_distribution(user_records),
                "asst_weight_dist": self.ltp.get_weight_distribution(asst_records),
            }
        if self.associations:
            result["associations"] = {
                "active_links": self.associations.num_active_links,
                "total_expansions": self.associations.total_expansions,
                "co_retrieval_pairs": len(self.associations.co_retrieval),
            }
        if self.inhibitor:
            result["inhibition"] = {
                "total_inhibitions": self.inhibitor.total_inhibitions,
            }
        if self.consolidator:
            result["consolidation"] = {
                "merges": self.consolidator.merges_performed,
                "abstractions": self.consolidator.abstractions_created,
            }
        return result
