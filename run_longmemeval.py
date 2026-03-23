"""
LongMemEval adapter for the gated-memory pipeline.

LongMemEval differs from LoCoMo:
  - Each of the 500 instances has its own chat history (haystack)
  - Turns are {role: "user"/"assistant"} instead of named speakers
  - ~40 sessions per instance, ~500+ turns each
  - Six question types, with their own judge prompts
  - One question per instance

Modes:
  naive       — store every turn, raw cosine retrieval
  enhanced    — multi-signal gating (surprise + temporal + entity)
  inhibition  — enhanced + belief revision inhibition at encoding time
  neuroplastic — enhanced + inhibition + consolidation
                 (LTP and associations are per-instance plumbing but
                  only meaningful when multiple questions share a memory
                  bank, i.e. LoCoMo. Here they run but have limited
                  impact since there's 1 question per instance.)
"""

import os
import json
import time
import argparse
from collections import Counter

import faiss
import numpy as np
from tqdm import tqdm
from jinja2 import Template
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

ANSWER_PROMPT = """You are a helpful chat assistant with long-term memory. You have access to memories from past conversations with the user.

# MEMORIES FROM PAST CONVERSATIONS:

User's messages:
{{user_memories}}

Assistant's messages:
{{assistant_memories}}

# INSTRUCTIONS:
1. Use the memories above to answer the user's question accurately.
2. Pay attention to timestamps — they indicate when each conversation happened.
3. If information was updated over time, use the most recent information.
4. If the question asks about time differences, calculate based on the timestamps.
5. Be concise and specific. Answer in 1-2 sentences.

Question (asked on {{question_date}}): {{question}}

Answer:"""

ANSWER_PROMPT_COT = """You are a helpful chat assistant with perfect long-term memory of all past conversations with the user.

# MEMORIES FROM PAST CONVERSATIONS:

User's messages:
{{user_memories}}

Assistant's messages:
{{assistant_memories}}

# TASK:
Answer the question below using ONLY the memories above. Follow these reasoning steps:

Step 1 — Identify every memory relevant to the question. List them.
Step 2 — If the question asks "how many" or to count/list items, enumerate each distinct item found in the memories before giving a total.
Step 3 — If information was updated over time (same topic, different dates), use the MOST RECENT version. State which date you are using.
Step 4 — If the question involves time calculations, write out the start date, end date, and arithmetic.
Step 5 — If the question asks about preferences, infer from the user's past behavior, stated opinions, and choices.
Step 6 — Give your final answer in 1-2 concise sentences.

Question (asked on {{question_date}}): {{question}}

Let me work through this:"""


class LongMemEvalSystem:
    """Runs LongMemEval with configurable memory strategies."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 30,
        answer_workers: int = 1,
        mode: str = "naive",
        gate_threshold: float = 0.2,
        gate_mode: str = "fixed",
        gate_metric: str = "nearest_neighbor",
        gate_warmup: int = 3,
        plasticity_config=None,
        use_cot: bool = False,
    ):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.top_k = top_k
        self.answer_workers = answer_workers
        self._openai_client = None
        self.answer_model = os.getenv("MODEL", "gpt-4o-mini")
        self.use_cot = use_cot

        self.mode = mode
        self.gate_threshold = gate_threshold
        self.gate_mode = gate_mode
        self.gate_metric = gate_metric
        self.gate_warmup = gate_warmup
        self.plasticity_config = plasticity_config

        if mode in ("enhanced", "inhibition", "neuroplastic"):
            from enhanced_gated_encoder import (
                EnhancedGatedEncoder, TemporalDetector, EntityTracker, MemoryRecord,
            )
            self._TemporalDetector = TemporalDetector
            self._EntityTracker = EntityTracker
            self._EnhancedGatedEncoder = EnhancedGatedEncoder
            self._MemoryRecord = MemoryRecord

        if mode == "neuroplastic":
            from neuroplastic_memory import NeuroplasticMemory, PlasticityConfig
            self._NeuroplasticMemory = NeuroplasticMemory
            self._PlasticityConfig = PlasticityConfig

    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    def build_memories(self, instance: dict) -> dict:
        sessions = instance["haystack_sessions"]
        dates = instance["haystack_dates"]

        if self.mode == "naive":
            return self._build_naive(sessions, dates)
        return self._build_gated(sessions, dates)

    def _build_naive(self, sessions, dates) -> dict:
        user_texts, asst_texts = [], []
        for sess, date in zip(sessions, dates):
            for turn in sess:
                text = f"{date} | {turn['role']}: {turn['content']}"
                if turn["role"] == "user":
                    user_texts.append(text)
                else:
                    asst_texts.append(text)

        return {
            "user_mems": user_texts,
            "asst_mems": asst_texts,
            "user_index": self._build_faiss_index(user_texts),
            "asst_index": self._build_faiss_index(asst_texts),
            "gate_stats": {
                "user": {"total_seen": len(user_texts), "total_stored": len(user_texts)},
                "assistant": {"total_seen": len(asst_texts), "total_stored": len(asst_texts)},
            },
        }

    def _build_gated(self, sessions, dates) -> dict:
        td = self._TemporalDetector()
        et = self._EntityTracker()
        gate_user = self._EnhancedGatedEncoder(
            threshold=self.gate_threshold, mode=self.gate_mode,
            metric=self.gate_metric, warmup=self.gate_warmup,
            temporal_detector=td, entity_tracker=et,
        )
        gate_asst = self._EnhancedGatedEncoder(
            threshold=self.gate_threshold, mode=self.gate_mode,
            metric=self.gate_metric, warmup=self.gate_warmup,
            temporal_detector=td, entity_tracker=et,
        )

        user_records, asst_records = [], []

        for sess, date in zip(sessions, dates):
            texts_batch = [f"{date} | {t['role']}: {t['content']}" for t in sess]
            raw_batch = [t["content"] for t in sess]
            embeddings = self.embedder.encode(
                texts_batch, show_progress_bar=False, normalize_embeddings=True
            )

            for turn, text, raw, emb in zip(sess, texts_batch, raw_batch, embeddings):
                role = turn["role"]
                gate = gate_user if role == "user" else gate_asst
                target = user_records if role == "user" else asst_records

                should_store, surprise, temporal, entity = gate.gate(emb, raw, role)
                if should_store:
                    rec = self._MemoryRecord(
                        text=text, embedding=emb, created_at=date,
                        surprise_score=surprise, temporal_salience=temporal,
                        entity_novelty=entity,
                    )
                    target.append(rec)

        npm = None
        if self.mode in ("inhibition", "neuroplastic"):
            if self.mode == "neuroplastic":
                npm = self._NeuroplasticMemory(self.plasticity_config)
                npm.apply_inhibition(user_records)
                npm.apply_inhibition(asst_records)
                user_records = npm.consolidate(user_records)
                asst_records = npm.consolidate(asst_records)
            else:
                self._apply_inhibition(user_records)
                self._apply_inhibition(asst_records)

        user_embs = [r.embedding for r in user_records]
        asst_embs = [r.embedding for r in asst_records]

        return {
            "user_mems": [r.text for r in user_records],
            "asst_mems": [r.text for r in asst_records],
            "user_records": user_records,
            "asst_records": asst_records,
            "user_index": self._build_faiss_index_from_embs(user_embs),
            "asst_index": self._build_faiss_index_from_embs(asst_embs),
            "neuroplastic": npm,
            "gate_stats": {
                "user": gate_user.stats.summary(),
                "assistant": gate_asst.stats.summary(),
            },
        }

    def _apply_inhibition(self, records: list):
        if len(records) < 2:
            return
        n = len(records)
        for i in range(n):
            if records[i].inhibition_weight >= 0.8:
                continue
            for j in range(i + 1, n):
                sim = float(np.dot(records[i].embedding, records[j].embedding))
                if sim > 0.85 and records[i].created_at != records[j].created_at:
                    records[i].inhibited_by = j
                    records[i].inhibition_weight = 0.8
                    break

    def _build_faiss_index(self, texts: list[str]) -> faiss.Index | None:
        if not texts:
            return None
        embs = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(np.array(embs, dtype=np.float32))
        return index

    def _build_faiss_index_from_embs(self, embeddings: list[np.ndarray]) -> faiss.Index | None:
        if not embeddings:
            return None
        arr = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(arr)
        return index

    def retrieve(self, query: str, mems: list[str], index: faiss.Index | None,
                 records: list | None = None, npm=None) -> tuple[list[str], list[int]]:
        """Returns (retrieved_texts, retrieved_indices)."""
        if index is None or not mems:
            return [], []
        k = min(self.top_k, len(mems))

        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)

        if npm and records:
            candidate_k = min(k * 3, len(records))
            raw_scores, indices = index.search(query_emb, candidate_k)
            scored = npm.score_candidates(query, raw_scores[0], indices[0], records, k)
            result_indices = [idx for _, idx in scored]
            texts = [records[i].text if i < len(records) else mems[i] for i in result_indices]
            return texts, result_indices

        if records and any(r.inhibition_weight > 0 or r.retrieval_weight != 1.0 for r in records):
            candidate_k = min(k * 3, len(records))
            raw_scores, indices = index.search(query_emb, candidate_k)

            from neuroplastic_memory import BeliefRevisionDetector
            asking_past = BeliefRevisionDetector.is_asking_about_past(query)

            weighted = []
            for score, idx in zip(raw_scores[0], indices[0]):
                if idx >= len(records):
                    continue
                rec = records[idx]
                eff_inhib = BeliefRevisionDetector.effective_inhibition(rec, asking_past)
                final = float(score) * rec.retrieval_weight * (1.0 - eff_inhib)
                weighted.append((final, int(idx)))

            weighted.sort(key=lambda x: x[0], reverse=True)
            result_indices = [idx for _, idx in weighted[:k]]
            return [records[i].text for i in result_indices], result_indices

        scores, indices = index.search(query_emb, k)
        result_indices = [int(i) for i in indices[0] if i < len(mems)]
        return [mems[i] for i in result_indices], result_indices

    def answer_question(self, question: str, question_date: str,
                        user_retrieved: list[str], asst_retrieved: list[str],
                        max_retries: int = 8) -> tuple[str, float]:
        prompt_text = ANSWER_PROMPT_COT if self.use_cot else ANSWER_PROMPT
        template = Template(prompt_text)
        prompt = template.render(
            user_memories=json.dumps(user_retrieved, indent=2),
            assistant_memories=json.dumps(asst_retrieved, indent=2),
            question=question,
            question_date=question_date,
        )

        for attempt in range(max_retries):
            try:
                t1 = time.time()
                resp = self.openai_client.chat.completions.create(
                    model=self.answer_model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip(), time.time() - t1
            except Exception as e:
                wait = min(2 ** attempt, 60)
                print(f"\n  API error (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  Waiting {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"Failed after {max_retries} retries")

    def process_instance(self, instance: dict, idx: int) -> dict:
        mem = self.build_memories(instance)

        user_stats = mem["gate_stats"]["user"]
        asst_stats = mem["gate_stats"]["assistant"]
        u_stored = user_stats.get("total_stored", user_stats.get("total_seen", "?"))
        u_seen = user_stats.get("total_seen", "?")
        a_stored = asst_stats.get("total_stored", asst_stats.get("total_seen", "?"))
        a_seen = asst_stats.get("total_seen", "?")

        user_records = mem.get("user_records")
        asst_records = mem.get("asst_records")
        npm = mem.get("neuroplastic")

        user_retrieved, user_ret_idx = self.retrieve(
            instance["question"], mem["user_mems"], mem["user_index"],
            user_records, npm,
        )
        asst_retrieved, asst_ret_idx = self.retrieve(
            instance["question"], mem["asst_mems"], mem["asst_index"],
            asst_records, npm,
        )

        response, response_time = self.answer_question(
            instance["question"],
            instance["question_date"],
            user_retrieved,
            asst_retrieved,
        )

        extra = {}
        if npm:
            extra["plasticity_stats"] = npm.stats_summary(
                user_records or [], asst_records or []
            )

        return {
            "question_id": instance["question_id"],
            "question_type": instance["question_type"],
            "question": instance["question"],
            "answer": instance["answer"],
            "hypothesis": response,
            "response_time": response_time,
            "user_stored": u_stored,
            "user_seen": u_seen,
            "asst_stored": a_stored,
            "asst_seen": a_seen,
            **extra,
        }


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark")
    parser.add_argument("--dataset", type=str,
                        default="LongMemEval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--mode", type=str, default="naive",
                        choices=["naive", "enhanced", "inhibition", "neuroplastic"])
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--gate_mode", type=str, default="fixed")
    parser.add_argument("--gate_metric", type=str, default="nearest_neighbor")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--answer_workers", type=int, default=1)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--cot", action="store_true", default=False,
                        help="Use chain-of-thought answer prompt")

    # Plasticity flags (only for neuroplastic mode)
    parser.add_argument("--enable_ltp", action="store_true", default=True)
    parser.add_argument("--no_ltp", dest="enable_ltp", action="store_false")
    parser.add_argument("--enable_associations", action="store_true", default=True)
    parser.add_argument("--no_associations", dest="enable_associations", action="store_false")
    parser.add_argument("--enable_inhibition", action="store_true", default=True)
    parser.add_argument("--no_inhibition", dest="enable_inhibition", action="store_false")
    parser.add_argument("--enable_consolidation", action="store_true", default=True)
    parser.add_argument("--no_consolidation", dest="enable_consolidation", action="store_false")

    args = parser.parse_args()

    if args.output is None:
        suffix = "_cot" if args.cot else ""
        args.output = f"results/lme_{args.mode}{suffix}.json"

    plasticity_config = None
    if args.mode == "neuroplastic":
        from neuroplastic_memory import PlasticityConfig
        plasticity_config = PlasticityConfig(
            enable_ltp=args.enable_ltp,
            enable_associations=args.enable_associations,
            enable_inhibition=args.enable_inhibition,
            enable_consolidation=args.enable_consolidation,
        )

    print(f"=== LongMemEval Benchmark ===")
    print(f"  mode={args.mode}, top_k={args.top_k}")
    if args.mode != "naive":
        print(f"  threshold={args.threshold}, gate_mode={args.gate_mode}")
    if args.mode == "neuroplastic":
        flags = []
        if args.enable_ltp: flags.append("LTP")
        if args.enable_associations: flags.append("Assoc")
        if args.enable_inhibition: flags.append("Inhib")
        if args.enable_consolidation: flags.append("Consol")
        print(f"  plasticity: {' + '.join(flags)}")
    print(f"  output={args.output}")

    with open(args.dataset) as f:
        data = json.load(f)

    if args.max_instances:
        data = data[:args.max_instances]
        print(f"  (limited to first {args.max_instances} instances)")

    print(f"  {len(data)} instances to process")

    system = LongMemEvalSystem(
        top_k=args.top_k,
        answer_workers=args.answer_workers,
        mode=args.mode,
        gate_threshold=args.threshold,
        gate_mode=args.gate_mode,
        gate_metric=args.gate_metric,
        gate_warmup=args.warmup,
        plasticity_config=plasticity_config,
        use_cot=args.cot,
    )

    # Resume support
    results = []
    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        done_ids = {r["question_id"] for r in results}
        print(f"  Resuming — {len(done_ids)} instances already done")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for idx, instance in enumerate(tqdm(data, desc="Processing instances")):
        if instance["question_id"] in done_ids:
            continue

        try:
            result = system.process_instance(instance, idx)
            results.append(result)

            if (len(results) % 5 == 0) or idx == len(data) - 1:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)

            qtype = instance["question_type"][:12]
            print(f"  [{idx}] {qtype:>12s} | user={result['user_stored']}/{result['user_seen']} "
                  f"asst={result['asst_stored']}/{result['asst_seen']} | "
                  f"{result['response_time']:.1f}s")
        except Exception as e:
            print(f"  [{idx}] ERROR: {e}")
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    type_counts = Counter(r["question_type"] for r in results)
    total_user_stored = sum(r.get("user_stored", 0) for r in results if isinstance(r.get("user_stored"), int))
    total_user_seen = sum(r.get("user_seen", 0) for r in results if isinstance(r.get("user_seen"), int))

    print(f"\n=== Summary ===")
    print(f"Instances processed: {len(results)}")
    for k, v in sorted(type_counts.items()):
        print(f"  {k}: {v}")
    if total_user_seen > 0:
        print(f"Total user memories: {total_user_stored}/{total_user_seen} "
              f"({1 - total_user_stored/total_user_seen:.1%} compressed)")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
