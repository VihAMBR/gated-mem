#!/usr/bin/env python3
"""Pre-compute embeddings for LongMemEval experiment suite.

Embeds all turns for a subset of instances with both all-MiniLM-L6-v2 and
BAAI/bge-small-en-v1.5, builds FAISS indices, and saves everything to a
pickle cache. Experiments then load this cache instead of re-embedding.
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


def precompute(data, embedder, model_name, show_progress=True):
    dim = embedder.get_sentence_embedding_dimension()
    cache = []

    iterator = tqdm(data, desc=f"Embedding [{model_name}]") if show_progress else data
    for instance in iterator:
        sessions = instance["haystack_sessions"]
        dates = instance["haystack_dates"]

        user_texts, asst_texts = [], []
        for sess, date in zip(sessions, dates):
            for turn in sess:
                text = f"{date} | {turn['role']}: {turn['content']}"
                if turn["role"] == "user":
                    user_texts.append(text)
                else:
                    asst_texts.append(text)

        user_embs = embedder.encode(user_texts, show_progress_bar=False, normalize_embeddings=True) if user_texts else np.array([])
        asst_embs = embedder.encode(asst_texts, show_progress_bar=False, normalize_embeddings=True) if asst_texts else np.array([])

        user_index = None
        if len(user_embs) > 0:
            user_index = faiss.IndexFlatIP(dim)
            user_index.add(np.array(user_embs, dtype=np.float32))

        asst_index = None
        if len(asst_embs) > 0:
            asst_index = faiss.IndexFlatIP(dim)
            asst_index.add(np.array(asst_embs, dtype=np.float32))

        # Multi-turn windows
        mt_user_texts, mt_asst_texts = [], []
        for sess, date in zip(sessions, dates):
            turns = [(t["role"], f"{date} | {t['role']}: {t['content']}") for t in sess]
            for i in range(len(turns)):
                window = [text for _, text in turns[max(0, i-1):i+2]]
                combined = " ||| ".join(window)
                if turns[i][0] == "user":
                    mt_user_texts.append(combined)
                else:
                    mt_asst_texts.append(combined)

        mt_user_embs = embedder.encode(mt_user_texts, show_progress_bar=False, normalize_embeddings=True) if mt_user_texts else np.array([])
        mt_asst_embs = embedder.encode(mt_asst_texts, show_progress_bar=False, normalize_embeddings=True) if mt_asst_texts else np.array([])

        mt_user_index = None
        if len(mt_user_embs) > 0:
            mt_user_index = faiss.IndexFlatIP(dim)
            mt_user_index.add(np.array(mt_user_embs, dtype=np.float32))
        mt_asst_index = None
        if len(mt_asst_embs) > 0:
            mt_asst_index = faiss.IndexFlatIP(dim)
            mt_asst_index.add(np.array(mt_asst_embs, dtype=np.float32))

        # Recency positions
        total_turns = sum(len(s) for s in sessions)
        user_pos, asst_pos = [], []
        pos = 0
        for sess, date in zip(sessions, dates):
            for turn in sess:
                r = pos / max(total_turns - 1, 1)
                if turn["role"] == "user":
                    user_pos.append(r)
                else:
                    asst_pos.append(r)
                pos += 1

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
            "mt_user_mems": mt_user_texts,
            "mt_asst_mems": mt_asst_texts,
            "mt_user_index": mt_user_index,
            "mt_asst_index": mt_asst_index,
            "user_positions": user_pos,
            "asst_positions": asst_pos,
            "dim": dim,
        })

    return cache


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
    cache_minilm = precompute(data, embedder, "all-MiniLM-L6-v2")
    del embedder

    embedder_bge = SentenceTransformer("BAAI/bge-small-en-v1.5")
    cache_bge = precompute(data, embedder_bge, "bge-small-en-v1.5")
    del embedder_bge

    result = {"minilm": cache_minilm, "bge": cache_bge}

    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    elapsed = time.time() - t0
    print(f"\nSaved {len(cache_minilm)} instances to {args.output}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
