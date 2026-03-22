"""
Surprise-gated memory system for the LoCoMo benchmark.

Same as the naive baseline, except we DON'T store every conversation turn.
Instead, each turn is passed through a SurpriseGatedEncoder that decides
whether the message is novel enough to store. This tests whether selective
memory (storing less, but more informative data) beats brute-force storage.
"""

import os
import json
import time
import argparse
from collections import defaultdict

import faiss
import numpy as np
from tqdm import tqdm
from jinja2 import Template
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from surprise_gated_encoder import SurpriseGatedEncoder, GateStats

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


class GatedMemorySystem:

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 30,
        gate_threshold: float = 0.3,
        gate_mode: str = "fixed",
        gate_metric: str = "nearest_neighbor",
        gate_warmup: int = 3,
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

    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    def build_memories(self, conversation: dict) -> dict:
        """
        Walk through every session chronologically, and for each turn:
          1. Embed it
          2. Ask the surprise gate: is this novel?
          3. Only store it if yes

        Each speaker gets their own gate — what's surprising for speaker A
        depends on what A has said, not what B said.
        """
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        gate_a = SurpriseGatedEncoder(
            threshold=self.gate_threshold,
            mode=self.gate_mode,
            metric=self.gate_metric,
            warmup=self.gate_warmup,
        )
        gate_b = SurpriseGatedEncoder(
            threshold=self.gate_threshold,
            mode=self.gate_mode,
            metric=self.gate_metric,
            warmup=self.gate_warmup,
        )

        speaker_a_mems: list[str] = []
        speaker_b_mems: list[str] = []
        speaker_a_embs: list[np.ndarray] = []
        speaker_b_embs: list[np.ndarray] = []

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
                embeddings = self.embedder.encode(
                    texts, show_progress_bar=False, normalize_embeddings=True
                )

                for turn, text, emb in zip(turns, texts, embeddings):
                    if turn["speaker"] == speaker_a:
                        should_store, _ = gate_a.gate(emb)
                        if should_store:
                            speaker_a_mems.append(text)
                            speaker_a_embs.append(emb)
                    else:
                        should_store, _ = gate_b.gate(emb)
                        if should_store:
                            speaker_b_mems.append(text)
                            speaker_b_embs.append(emb)

            session_idx += 1

        speaker_a_index = self._build_faiss_index_from_embs(speaker_a_embs)
        speaker_b_index = self._build_faiss_index_from_embs(speaker_b_embs)

        return {
            "speaker_a_name": speaker_a,
            "speaker_b_name": speaker_b,
            "speaker_a_memories": speaker_a_mems,
            "speaker_b_memories": speaker_b_mems,
            "speaker_a_index": speaker_a_index,
            "speaker_b_index": speaker_b_index,
            "gate_a_stats": gate_a.stats,
            "gate_b_stats": gate_b.stats,
        }

    def _build_faiss_index_from_embs(self, embeddings: list[np.ndarray]) -> faiss.Index | None:
        if not embeddings:
            return None
        emb_array = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(emb_array)
        return index

    def retrieve(self, query: str, memories: list[str], index: faiss.Index | None, top_k: int | None = None) -> list[str]:
        if index is None or not memories:
            return []
        k = min(top_k or self.top_k, len(memories))
        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(query_emb, k)
        return [memories[i] for i in indices[0] if i < len(memories)]

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

        gate_info = {
            "speaker_a": {
                "total_seen": mem["gate_a_stats"].total_seen,
                "total_stored": mem["gate_a_stats"].total_stored,
                "compression_ratio": mem["gate_a_stats"].compression_ratio,
                "mean_surprise": mem["gate_a_stats"].mean_surprise,
            },
            "speaker_b": {
                "total_seen": mem["gate_b_stats"].total_seen,
                "total_stored": mem["gate_b_stats"].total_stored,
                "compression_ratio": mem["gate_b_stats"].compression_ratio,
                "mean_surprise": mem["gate_b_stats"].mean_surprise,
            },
        }

        print(f"    Gate: A stored {gate_info['speaker_a']['total_stored']}/{gate_info['speaker_a']['total_seen']} "
              f"({gate_info['speaker_a']['compression_ratio']:.0%} compressed), "
              f"B stored {gate_info['speaker_b']['total_stored']}/{gate_info['speaker_b']['total_seen']} "
              f"({gate_info['speaker_b']['compression_ratio']:.0%} compressed)")

        results = []
        for qa in tqdm(qa_pairs, desc=f"  Conv {conv_idx} questions", leave=False):
            question = qa["question"]
            answer = qa.get("answer", "")
            category = qa["category"]

            if int(category) == 5:
                continue

            a_retrieved = self.retrieve(question, mem["speaker_a_memories"], mem["speaker_a_index"])
            b_retrieved = self.retrieve(question, mem["speaker_b_memories"], mem["speaker_b_index"])

            response, response_time = self.answer_question(
                question,
                mem["speaker_a_name"],
                mem["speaker_b_name"],
                a_retrieved,
                b_retrieved,
            )

            results.append({
                "question": question,
                "answer": answer,
                "category": category,
                "response": response,
                "num_speaker_a_retrieved": len(a_retrieved),
                "num_speaker_b_retrieved": len(b_retrieved),
                "response_time": response_time,
            })

        return results, gate_info


def main():
    parser = argparse.ArgumentParser(description="Run surprise-gated memory on LoCoMo")
    parser.add_argument("--dataset", type=str, default="dataset/locomo10.json")
    parser.add_argument("--output", type=str, default="results/gated_results.json")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--mode", type=str, default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--metric", type=str, default="nearest_neighbor", choices=["nearest_neighbor", "centroid"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max_convs", type=int, default=None, help="Only process first N conversations (for quick testing)")
    args = parser.parse_args()

    tag = f"t{args.threshold}_{args.mode}_{args.metric}"
    if args.output == "results/gated_results.json":
        args.output = f"results/gated_{tag}.json"

    print(f"=== Surprise-Gated Memory ===")
    print(f"  threshold={args.threshold}, mode={args.mode}, metric={args.metric}, warmup={args.warmup}")
    print(f"  output={args.output}")

    with open(args.dataset) as f:
        data = json.load(f)

    if args.max_convs:
        data = data[:args.max_convs]
        print(f"  (limited to first {args.max_convs} conversations)")

    system = GatedMemorySystem(
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
        gate_threshold=args.threshold,
        gate_mode=args.mode,
        gate_metric=args.metric,
        gate_warmup=args.warmup,
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

    # Print summary
    total_seen = sum(g["speaker_a"]["total_seen"] + g["speaker_b"]["total_seen"] for g in all_gate_info.values())
    total_stored = sum(g["speaker_a"]["total_stored"] + g["speaker_b"]["total_stored"] for g in all_gate_info.values())
    print(f"\n=== Summary ===")
    print(f"Total memories: {total_stored}/{total_seen} stored ({1 - total_stored/total_seen:.1%} compressed)")
    print(f"Questions answered: {total_questions}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
