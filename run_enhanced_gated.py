"""
Enhanced gated memory system for the LoCoMo benchmark.

Builds on the surprise-gated baseline with two bypass mechanisms:
  - Temporal bypass: messages with dates/times/temporal keywords skip the gate
  - Entity novelty: messages introducing new named entities skip the gate

Retrieval uses weighted scoring:
  final_score = cosine_sim * retrieval_weight * (1 - inhibition_weight)

Currently all weights are neutral (1.0 / 0.0), so this is functionally
identical to raw cosine — but the plumbing is in place for future experiments
(decay, interference, consolidation).
"""

import os
import json
import time
import argparse
import concurrent.futures
from collections import defaultdict

import faiss
import numpy as np
from tqdm import tqdm
from jinja2 import Template
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from enhanced_gated_encoder import (
    EnhancedGatedEncoder,
    EnhancedGateStats,
    EntityTracker,
    TemporalDetector,
    MemoryRecord,
)

load_dotenv()

ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_name}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_name}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """


class EnhancedMemorySystem:

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 30,
        gate_threshold: float = 0.3,
        gate_mode: str = "fixed",
        gate_metric: str = "nearest_neighbor",
        gate_warmup: int = 3,
        answer_workers: int = 1,
    ):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.top_k = top_k
        self._openai_client = None
        self.answer_model = os.getenv("MODEL", "gpt-4o-mini")

        self.gate_threshold = gate_threshold
        self.gate_mode = gate_mode
        self.gate_metric = gate_metric
        self.gate_warmup = gate_warmup
        self.answer_workers = answer_workers

        self.temporal_detector = TemporalDetector()
        self.entity_tracker = EntityTracker()

    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    def build_memories(self, conversation: dict) -> dict:
        """
        Walk through sessions chronologically. For each turn, the enhanced
        gate decides whether to store based on surprise OR temporal markers
        OR novel entities. Each stored turn becomes a MemoryRecord.
        """
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        self.entity_tracker.reset()

        gate_a = EnhancedGatedEncoder(
            threshold=self.gate_threshold,
            mode=self.gate_mode,
            metric=self.gate_metric,
            warmup=self.gate_warmup,
            temporal_detector=self.temporal_detector,
            entity_tracker=self.entity_tracker,
        )
        gate_b = EnhancedGatedEncoder(
            threshold=self.gate_threshold,
            mode=self.gate_mode,
            metric=self.gate_metric,
            warmup=self.gate_warmup,
            temporal_detector=self.temporal_detector,
            entity_tracker=self.entity_tracker,
        )

        records_a: list[MemoryRecord] = []
        records_b: list[MemoryRecord] = []

        session_idx = 1
        while True:
            session_key = f"session_{session_idx}"
            date_key = f"{session_key}_date_time"
            if date_key not in conversation:
                break

            timestamp = conversation[date_key]
            turns = conversation.get(session_key)

            if turns and isinstance(turns, list):
                texts = [f"{timestamp} | {t['speaker']}: {t['text']}" for t in turns]
                raw_texts = [t["text"] for t in turns]
                embeddings = self.embedder.encode(
                    texts, show_progress_bar=False, normalize_embeddings=True
                )

                for turn, text, raw_text, emb in zip(turns, texts, raw_texts, embeddings):
                    speaker = turn["speaker"]
                    gate = gate_a if speaker == speaker_a else gate_b
                    target = records_a if speaker == speaker_a else records_b

                    should_store, surprise, temporal, entity = gate.gate(emb, raw_text, speaker)

                    if should_store:
                        record = MemoryRecord(
                            text=text,
                            embedding=emb,
                            created_at=timestamp,
                            surprise_score=surprise,
                            temporal_salience=temporal,
                            entity_novelty=entity,
                        )
                        target.append(record)

            session_idx += 1

        index_a = self._build_faiss_index([r.embedding for r in records_a])
        index_b = self._build_faiss_index([r.embedding for r in records_b])

        return {
            "speaker_a_name": speaker_a,
            "speaker_b_name": speaker_b,
            "records_a": records_a,
            "records_b": records_b,
            "index_a": index_a,
            "index_b": index_b,
            "gate_a_stats": gate_a.stats,
            "gate_b_stats": gate_b.stats,
        }

    def _build_faiss_index(self, embeddings: list[np.ndarray]) -> faiss.Index | None:
        if not embeddings:
            return None
        emb_array = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(emb_array)
        return index

    def retrieve(
        self,
        query: str,
        records: list[MemoryRecord],
        index: faiss.Index | None,
        top_k: int | None = None,
    ) -> list[str]:
        """
        Weighted retrieval:
          1. Get top_k*3 candidates from FAISS (raw cosine)
          2. Re-score: cosine_sim * retrieval_weight * (1 - inhibition_weight)
          3. Return top_k texts
        """
        if index is None or not records:
            return []
        k = top_k or self.top_k
        candidate_k = min(k * 3, len(records))

        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        raw_scores, indices = index.search(query_emb, candidate_k)

        weighted = []
        for score, idx in zip(raw_scores[0], indices[0]):
            if idx >= len(records):
                continue
            rec = records[idx]
            final_score = score * rec.retrieval_weight * (1.0 - rec.inhibition_weight)
            weighted.append((final_score, idx))

        weighted.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in weighted[:k]]

        return [records[i].text for i in top_indices]

    def answer_question(
        self,
        question: str,
        speaker_a_name: str,
        speaker_b_name: str,
        speaker_a_retrieved: list[str],
        speaker_b_retrieved: list[str],
        max_retries: int = 8,
    ) -> tuple[str, float]:
        template = Template(ANSWER_PROMPT)
        prompt = template.render(
            speaker_1_name=speaker_a_name,
            speaker_2_name=speaker_b_name,
            speaker_1_memories=json.dumps(speaker_a_retrieved, indent=2),
            speaker_2_memories=json.dumps(speaker_b_retrieved, indent=2),
            question=question,
        )

        for attempt in range(max_retries):
            try:
                t1 = time.time()
                response = self.openai_client.chat.completions.create(
                    model=self.answer_model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.0,
                )
                t2 = time.time()
                return response.choices[0].message.content.strip(), t2 - t1
            except Exception as e:
                wait = min(2 ** attempt, 60)
                print(f"\n  Rate limit/error (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
        raise RuntimeError(f"Failed after {max_retries} retries")

    def process_conversation(self, item: dict, conv_idx: int) -> tuple[list[dict], dict]:
        conversation = item["conversation"]
        qa_pairs = item["qa"]

        mem = self.build_memories(conversation)

        stats_a = mem["gate_a_stats"].summary()
        stats_b = mem["gate_b_stats"].summary()
        gate_info = {"speaker_a": stats_a, "speaker_b": stats_b}

        print(f"    Gate A: {stats_a['total_stored']}/{stats_a['total_seen']} stored "
              f"({stats_a['compression_ratio']:.0%} compressed) — "
              f"surprise={stats_a['stored_by_surprise']}, "
              f"temporal={stats_a['stored_by_temporal']}, "
              f"entity={stats_a['stored_by_entity']}, "
              f"warmup={stats_a['stored_by_warmup']}")
        print(f"    Gate B: {stats_b['total_stored']}/{stats_b['total_seen']} stored "
              f"({stats_b['compression_ratio']:.0%} compressed) — "
              f"surprise={stats_b['stored_by_surprise']}, "
              f"temporal={stats_b['stored_by_temporal']}, "
              f"entity={stats_b['stored_by_entity']}, "
              f"warmup={stats_b['stored_by_warmup']}")

        filtered_qas = [qa for qa in qa_pairs if int(qa["category"]) != 5]

        def _process_qa(qa):
            question = qa["question"]
            a_retrieved = self.retrieve(question, mem["records_a"], mem["index_a"])
            b_retrieved = self.retrieve(question, mem["records_b"], mem["index_b"])
            response, response_time = self.answer_question(
                question, mem["speaker_a_name"], mem["speaker_b_name"],
                a_retrieved, b_retrieved,
            )
            return {
                "question": question,
                "answer": qa.get("answer", ""),
                "category": qa["category"],
                "response": response,
                "num_speaker_a_retrieved": len(a_retrieved),
                "num_speaker_b_retrieved": len(b_retrieved),
                "response_time": response_time,
            }

        results = []
        if self.answer_workers <= 1:
            for qa in tqdm(filtered_qas, desc=f"  Conv {conv_idx} questions", leave=False):
                results.append(_process_qa(qa))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.answer_workers) as pool:
                futures = {pool.submit(_process_qa, qa): i for i, qa in enumerate(filtered_qas)}
                pbar = tqdm(total=len(filtered_qas), desc=f"  Conv {conv_idx} questions", leave=False)
                ordered = [None] * len(filtered_qas)
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    ordered[idx] = future.result()
                    pbar.update(1)
                pbar.close()
                results = ordered

        return results, gate_info


def main():
    parser = argparse.ArgumentParser(description="Run enhanced gated memory on LoCoMo")
    parser.add_argument("--dataset", type=str, default="dataset/locomo10.json")
    parser.add_argument("--output", type=str, default="results/enhanced_gated_results.json")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--mode", type=str, default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--metric", type=str, default="nearest_neighbor", choices=["nearest_neighbor", "centroid"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--answer_workers", type=int, default=1, help="Parallel workers for question answering")
    parser.add_argument("--max_convs", type=int, default=None, help="Only process first N conversations")
    args = parser.parse_args()

    tag = f"t{args.threshold}_{args.mode}_{args.metric}"
    if args.output == "results/enhanced_gated_results.json":
        args.output = f"results/enhanced_{tag}.json"

    print(f"=== Enhanced Gated Memory ===")
    print(f"  threshold={args.threshold}, mode={args.mode}, metric={args.metric}, warmup={args.warmup}")
    print(f"  bypasses: temporal=ON, entity_novelty=ON")
    print(f"  retrieval: weighted (cosine * weight * (1 - inhibition))")
    print(f"  output={args.output}")

    with open(args.dataset) as f:
        data = json.load(f)

    if args.max_convs:
        data = data[:args.max_convs]
        print(f"  (limited to first {args.max_convs} conversations)")

    system = EnhancedMemorySystem(
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
        gate_threshold=args.threshold,
        gate_mode=args.mode,
        gate_metric=args.metric,
        gate_warmup=args.warmup,
        answer_workers=args.answer_workers,
    )

    all_results = defaultdict(list)
    all_gate_info = {}

    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        if "gate_info" in existing:
            all_gate_info = existing["gate_info"]
        results_data = {k: v for k, v in existing.items() if k != "gate_info"}
        for k, v in results_data.items():
            all_results[int(k)] = v
        done = sum(len(v) for v in all_results.values())
        print(f"Resuming — {len(all_results)} conversations ({done} questions) already done.")

    total_questions = sum(len(v) for v in all_results.values())

    for idx, item in enumerate(tqdm(data, desc="Processing conversations")):
        if idx in all_results and len(all_results[idx]) > 0:
            print(f"  Conv {idx}: already done ({len(all_results[idx])} questions), skipping")
            continue

        results, gate_info = system.process_conversation(item, idx)
        all_results[idx] = results
        all_gate_info[str(idx)] = gate_info
        total_questions += len(results)

        save_data = dict(all_results)
        save_data["gate_info"] = all_gate_info
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"  Conv {idx}: {len(results)} questions answered")

    save_data = dict(all_results)
    save_data["gate_info"] = all_gate_info
    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2)

    total_seen = sum(
        g["speaker_a"]["total_seen"] + g["speaker_b"]["total_seen"]
        for g in all_gate_info.values()
    )
    total_stored = sum(
        g["speaker_a"]["total_stored"] + g["speaker_b"]["total_stored"]
        for g in all_gate_info.values()
    )
    total_temporal = sum(
        g["speaker_a"]["stored_by_temporal"] + g["speaker_b"]["stored_by_temporal"]
        for g in all_gate_info.values()
    )
    total_entity = sum(
        g["speaker_a"]["stored_by_entity"] + g["speaker_b"]["stored_by_entity"]
        for g in all_gate_info.values()
    )

    print(f"\n=== Summary ===")
    print(f"Total memories: {total_stored}/{total_seen} stored ({1 - total_stored/total_seen:.1%} compressed)")
    print(f"  Bypassed by temporal: {total_temporal}")
    print(f"  Bypassed by entity:   {total_entity}")
    print(f"Questions answered: {total_questions}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
