#!/usr/bin/env python3
"""Pre-compute MiniLM embeddings for LongMemEval experiment suite.

Embeds all turns for a stratified subset with all-MiniLM-L6-v2, builds FAISS
indices, and saves to a pickle cache. Experiments load this instead of
re-embedding.

BGE and multi-turn experiments compute their own embeddings (different model /
different chunking) and don't use this cache.
"""

import json
import pickle
import time
import argparse

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from run_lme_experiments import select_stratified_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="LongMemEval/data/longmemeval_s_cleaned.json")
    parser.add_argument("--max_instances", type=int, default=100)
    parser.add_argument("--output", default="results/lme_embedding_cache.pkl")
    args = parser.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)

    if args.max_instances < len(data):
        data = select_stratified_subset(data, args.max_instances)
    print(f"Pre-computing embeddings for {len(data)} instances")

    t0 = time.time()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    dim = embedder.get_sentence_embedding_dimension()

    cache = []
    for instance in tqdm(data, desc="Embedding"):
        sessions = instance["haystack_sessions"]
        dates = instance["haystack_dates"]

        user_texts, asst_texts = [], []
        user_pos, asst_pos = [], []
        total_turns = sum(len(s) for s in sessions)
        pos = 0

        for sess, date in zip(sessions, dates):
            for turn in sess:
                text = f"{date} | {turn['role']}: {turn['content']}"
                recency = pos / max(total_turns - 1, 1)
                if turn["role"] == "user":
                    user_texts.append(text)
                    user_pos.append(recency)
                else:
                    asst_texts.append(text)
                    asst_pos.append(recency)
                pos += 1

        user_embs = embedder.encode(user_texts, show_progress_bar=False, normalize_embeddings=True) if user_texts else np.zeros((0, dim), dtype=np.float32)
        asst_embs = embedder.encode(asst_texts, show_progress_bar=False, normalize_embeddings=True) if asst_texts else np.zeros((0, dim), dtype=np.float32)

        user_index = None
        if len(user_embs) > 0:
            user_index = faiss.IndexFlatIP(dim)
            user_index.add(np.array(user_embs, dtype=np.float32))

        asst_index = None
        if len(asst_embs) > 0:
            asst_index = faiss.IndexFlatIP(dim)
            asst_index.add(np.array(asst_embs, dtype=np.float32))

        cache.append({
            "question_id": instance["question_id"],
            "question_type": instance["question_type"],
            "question": instance["question"],
            "answer": instance["answer"],
            "question_date": instance["question_date"],
            "user_mems": user_texts,
            "asst_mems": asst_texts,
            "user_embs": user_embs,
            "asst_embs": asst_embs,
            "user_index": user_index,
            "asst_index": asst_index,
            "user_positions": user_pos,
            "asst_positions": asst_pos,
            "dim": dim,
            "haystack_sessions": instance["haystack_sessions"],
            "haystack_dates": instance["haystack_dates"],
        })

    with open(args.output, "wb") as f:
        pickle.dump(cache, f)

    elapsed = time.time() - t0
    print(f"\nSaved {len(cache)} instances to {args.output}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
