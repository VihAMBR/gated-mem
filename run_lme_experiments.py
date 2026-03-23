#!/usr/bin/env python3
"""
LongMemEval Experiment Suite — 8 targeted experiments to improve retrieval.

Uses pre-computed embeddings (from precompute_embeddings.py) to avoid redundant
encoding. Each experiment only does retrieval + API answering.

Experiments:
  1. rerank     — Cross-encoder reranking on FAISS top-50 candidates
  2. hyde       — Hypothetical Document Embeddings for retrieval
  3. decompose  — Query decomposition into sub-queries, merge results
  4. bge        — BAAI/bge-small-en-v1.5 embedding model upgrade
  5. multiturn  — 3-turn sliding window encoding
  6. topk50     — Increase top_k from 30 to 50
  7. recency    — Recency-boosted retrieval scoring
  8. cot        — Chain-of-thought answer prompt with category-aware reasoning

Usage:
    python precompute_embeddings.py --max_instances 100
    python run_lme_experiments.py --experiment rerank
    python run_lme_experiments.py --experiment cot
"""

import os
import json
import time
import pickle
import argparse
import random
from collections import defaultdict

import numpy as np
import faiss
from tqdm import tqdm
from jinja2 import Template
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EXPERIMENTS = ["rerank", "hyde", "decompose", "bge", "multiturn", "topk50", "recency", "cot"]

# ─── Answer Prompts ──────────────────────────────────────────────────────────

ANSWER_PROMPT_DEFAULT = """You are a helpful chat assistant with long-term memory. You have access to memories from past conversations with the user.

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


# ─── Stratified Subset Selection ──────────────────────────────────────────────

def select_stratified_subset(data, max_instances):
    by_type = defaultdict(list)
    for d in data:
        by_type[d["question_type"]].append(d)

    total = len(data)
    selected = []
    remaining = max_instances

    type_order = sorted(by_type.keys())
    for i, qtype in enumerate(type_order):
        pool = by_type[qtype]
        if i == len(type_order) - 1:
            n = remaining
        else:
            n = max(5, round(len(pool) / total * max_instances))
            n = min(n, len(pool), remaining)
        random.seed(42)
        selected.extend(random.sample(pool, min(n, len(pool))))
        remaining -= min(n, len(pool))

    random.seed(42)
    random.shuffle(selected)
    return selected[:max_instances]


# ─── Experiment Runner (cache-based) ─────────────────────────────────────────

class ExperimentRunner:
    def __init__(self, experiment: str, cache_path: str = "results/lme_embedding_cache.pkl"):
        self.experiment = experiment
        self._client = None
        self.answer_model = os.getenv("MODEL", "gpt-4o-mini")
        self.top_k = 30
        self.answer_template = ANSWER_PROMPT_DEFAULT

        with open(cache_path, "rb") as f:
            all_cache = pickle.load(f)

        if experiment == "bge":
            self.cache = all_cache["bge"]
        else:
            self.cache = all_cache["minilm"]

        self.cross_encoder = None
        self._embedder = None

        if experiment == "rerank":
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        if experiment in ("hyde", "decompose"):
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        if experiment == "bge":
            self._embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

        if experiment == "topk50":
            self.top_k = 50

        if experiment == "cot":
            self.answer_template = ANSWER_PROMPT_COT

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def _retrieve_standard(self, query, mems, index, k=None):
        if index is None or not mems:
            return []
        k = min(k or self.top_k, len(mems))

        if self._embedder:
            prefix = "Represent this sentence for searching relevant passages: " if self.experiment == "bge" else ""
            q_emb = self._embedder.encode([prefix + query], normalize_embeddings=True).astype(np.float32)
        else:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_default_embedder'):
                self._default_embedder = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = self._default_embedder.encode([query], normalize_embeddings=True).astype(np.float32)

        scores, indices = index.search(q_emb, k)
        return [mems[int(i)] for i in indices[0] if int(i) < len(mems)]

    def _get_query_embedding(self, query):
        if self._embedder:
            prefix = "Represent this sentence for searching relevant passages: " if self.experiment == "bge" else ""
            return self._embedder.encode([prefix + query], normalize_embeddings=True).astype(np.float32)
        if not hasattr(self, '_default_embedder'):
            self._default_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._default_embedder.encode([query], normalize_embeddings=True).astype(np.float32)

    def retrieve(self, query, entry):
        exp = self.experiment

        if exp == "multiturn":
            user_ret = self._retrieve_standard(query, entry["mt_user_mems"], entry["mt_user_index"])
            asst_ret = self._retrieve_standard(query, entry["mt_asst_mems"], entry["mt_asst_index"])
            return user_ret, asst_ret

        user_mems, asst_mems = entry["user_mems"], entry["asst_mems"]
        user_idx, asst_idx = entry["user_index"], entry["asst_index"]

        if exp == "rerank":
            return self._rerank_retrieve(query, user_mems, user_idx), \
                   self._rerank_retrieve(query, asst_mems, asst_idx)

        if exp == "hyde":
            return self._hyde_retrieve(query, user_mems, user_idx), \
                   self._hyde_retrieve(query, asst_mems, asst_idx)

        if exp == "decompose":
            return self._decompose_retrieve(query, user_mems, user_idx), \
                   self._decompose_retrieve(query, asst_mems, asst_idx)

        if exp == "recency":
            return self._recency_retrieve(query, user_mems, user_idx, entry["user_positions"]), \
                   self._recency_retrieve(query, asst_mems, asst_idx, entry["asst_positions"])

        return self._retrieve_standard(query, user_mems, user_idx), \
               self._retrieve_standard(query, asst_mems, asst_idx)

    def _rerank_retrieve(self, query, mems, index):
        if index is None or not mems:
            return []
        candidate_k = min(50, len(mems))
        q_emb = self._get_query_embedding(query)
        scores, indices = index.search(q_emb, candidate_k)
        candidates = [(mems[int(i)], int(i)) for i in indices[0] if int(i) < len(mems)]
        if not candidates:
            return []
        pairs = [(query, text) for text, _ in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        return [text for _, (text, _) in ranked[:self.top_k]]

    def _hyde_retrieve(self, query, mems, index):
        if index is None or not mems:
            return []
        hyde_prompt = (
            "You are helping a retrieval system find relevant conversation memories.\n"
            "Given the question below, write a short (1-2 sentence) hypothetical excerpt "
            "from a past conversation that would contain the answer.\n\n"
            f"Question: {query}\n\nHypothetical memory:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.answer_model,
                messages=[{"role": "user", "content": hyde_prompt}],
                temperature=0.0, max_tokens=100,
            )
            hyde_text = resp.choices[0].message.content.strip()
        except Exception:
            hyde_text = ""

        combined = f"{query} {hyde_text}" if hyde_text else query
        k = min(self.top_k, len(mems))
        q_emb = self._embedder.encode([combined], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(q_emb, k)
        return [mems[int(i)] for i in indices[0] if int(i) < len(mems)]

    def _decompose_retrieve(self, query, mems, index):
        if index is None or not mems:
            return []
        decompose_prompt = (
            "Break this question into 2-3 independent search queries that would help "
            "find relevant memories in a conversation history. Return one query per line. "
            "If the question is simple, return just the original question.\n\n"
            f"Question: {query}\n\nSearch queries:"
        )
        sub_queries = [query]
        try:
            resp = self.client.chat.completions.create(
                model=self.answer_model,
                messages=[{"role": "user", "content": decompose_prompt}],
                temperature=0.0, max_tokens=200,
            )
            for line in resp.choices[0].message.content.strip().split("\n"):
                line = line.strip().lstrip("0123456789.-) ")
                if line and len(line) > 10 and line != query:
                    sub_queries.append(line)
        except Exception:
            pass

        all_indices = set()
        per_k = max(10, self.top_k // len(sub_queries))
        for sq in sub_queries:
            k = min(per_k, len(mems))
            q_emb = self._embedder.encode([sq], normalize_embeddings=True).astype(np.float32)
            scores, indices = index.search(q_emb, k)
            for i in indices[0]:
                if int(i) < len(mems):
                    all_indices.add(int(i))

        if not all_indices:
            return []

        q_emb = self._embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        idx_list = list(all_indices)
        mem_embs = np.array([
            self._embedder.encode([mems[i]], normalize_embeddings=True)[0] for i in idx_list
        ], dtype=np.float32)
        sims = np.dot(mem_embs, q_emb.T).flatten()
        ranked = sorted(zip(sims, idx_list), reverse=True)
        return [mems[idx] for _, idx in ranked[:self.top_k]]

    def _recency_retrieve(self, query, mems, index, positions):
        if index is None or not mems:
            return []
        k = min(self.top_k, len(mems))
        candidate_k = min(k * 2, len(mems))
        q_emb = self._get_query_embedding(query)
        scores, indices = index.search(q_emb, candidate_k)

        recency_weight = 0.15
        weighted = []
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx >= len(mems):
                continue
            rec = positions[idx] if idx < len(positions) else 0.5
            final = float(score) * (1.0 + recency_weight * rec)
            weighted.append((final, idx))

        weighted.sort(key=lambda x: x[0], reverse=True)
        return [mems[idx] for _, idx in weighted[:k]]

    def answer(self, question, question_date, user_ret, asst_ret, max_retries=8):
        template = Template(self.answer_template)
        prompt = template.render(
            user_memories=json.dumps(user_ret, indent=2),
            assistant_memories=json.dumps(asst_ret, indent=2),
            question=question,
            question_date=question_date,
        )
        for attempt in range(max_retries):
            try:
                t1 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.answer_model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip(), time.time() - t1
            except Exception as e:
                wait = min(2 ** attempt, 60)
                print(f"\n  API error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(wait)
        raise RuntimeError(f"Failed after {max_retries} retries")

    def process(self, entry):
        user_ret, asst_ret = self.retrieve(entry["question"], entry)
        response, rtime = self.answer(
            entry["question"], entry["question_date"], user_ret, asst_ret
        )
        return {
            "question_id": entry["question_id"],
            "question_type": entry["question_type"],
            "question": entry["question"],
            "answer": entry["answer"],
            "hypothesis": response,
            "response_time": rtime,
            "user_stored": len(entry["user_mems"]),
            "user_seen": len(entry["user_mems"]),
            "asst_stored": len(entry["asst_mems"]),
            "asst_seen": len(entry["asst_mems"]),
        }


def run_experiment(experiment, cache_path, output_path):
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    runner = ExperimentRunner(experiment, cache_path)
    n = len(runner.cache)
    print(f"  {n} instances from cache")

    results = []
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = {r["question_id"] for r in results}
        print(f"  Resuming — {len(done_ids)} already done")

    for idx, entry in enumerate(tqdm(runner.cache, desc=f"[{experiment}]")):
        if entry["question_id"] in done_ids:
            continue
        try:
            result = runner.process(entry)
            results.append(result)

            if len(results) % 5 == 0 or idx == n - 1:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

            qtype = entry["question_type"][:12]
            print(f"  [{idx}] {qtype:>12s} | {result['response_time']:.1f}s")
        except Exception as e:
            print(f"  [{idx}] ERROR: {e}")
            import traceback; traceback.print_exc()
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Done: {len(results)} results saved")


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Experiment Suite")
    parser.add_argument("--experiment", type=str, required=True, choices=EXPERIMENTS)
    parser.add_argument("--cache", type=str, default="results/lme_embedding_cache.pkl")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"lme_exp_{args.experiment}.json")
    run_experiment(args.experiment, args.cache, output_path)


if __name__ == "__main__":
    main()
