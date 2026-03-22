"""
Naive baseline memory system for the LoCoMo benchmark.

Strategy:
  1. For each conversation, flatten every turn into a timestamped memory string
  2. Embed all memories with sentence-transformers (all-MiniLM-L6-v2)
  3. Store embeddings in a FAISS flat index
  4. At query time, retrieve top-k most similar memories
  5. Feed retrieved memories + question into GPT-4o-mini using Mem0's answer prompt
  6. Collect answers for evaluation
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

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt — identical to Mem0's ANSWER_PROMPT from the benchmark repo.
# Using the same prompt is essential for comparable scores.
# ---------------------------------------------------------------------------
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


class NaiveMemoryBaseline:
    """
    Simplest possible memory system:
    - Every conversation turn becomes a memory (no summarization / dedup / extraction)
    - Embedded with a local sentence-transformer
    - Stored in a flat FAISS index (brute-force cosine similarity)
    - Top-k retrieval at query time
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", top_k: int = 30):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.top_k = top_k
        self._openai_client = None
        self.answer_model = os.getenv("MODEL", "gpt-4o-mini")

    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    # ------------------------------------------------------------------
    # Step 1 & 2: Turn a conversation into (text, embedding) memories
    # ------------------------------------------------------------------
    def build_memories(self, conversation: dict) -> dict:
        """
        Walk through every session in the conversation and produce one memory
        per conversation turn, formatted as:
            "<timestamp> | <Speaker>: <text>"

        We keep two separate memory banks — one per speaker — matching how
        Mem0 and Memobase organise memories. This lets the answer prompt
        reference "memories for user X" and "memories for user Y" separately.

        Returns dict with keys:
            speaker_a_name, speaker_b_name,
            speaker_a_memories (list[str]), speaker_b_memories (list[str]),
            speaker_a_index (faiss.Index), speaker_b_index (faiss.Index)
        """
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_mems: list[str] = []
        speaker_b_mems: list[str] = []

        session_idx = 1
        while True:
            session_key = f"session_{session_idx}"
            date_key = f"{session_key}_date_time"

            if date_key not in conversation:
                break

            timestamp = conversation[date_key]
            turns = conversation.get(session_key)

            if turns and isinstance(turns, list):
                for turn in turns:
                    memory_text = f"{timestamp} | {turn['speaker']}: {turn['text']}"
                    if turn["speaker"] == speaker_a:
                        speaker_a_mems.append(memory_text)
                    else:
                        speaker_b_mems.append(memory_text)

            session_idx += 1

        speaker_a_index = self._build_faiss_index(speaker_a_mems)
        speaker_b_index = self._build_faiss_index(speaker_b_mems)

        return {
            "speaker_a_name": speaker_a,
            "speaker_b_name": speaker_b,
            "speaker_a_memories": speaker_a_mems,
            "speaker_b_memories": speaker_b_mems,
            "speaker_a_index": speaker_a_index,
            "speaker_b_index": speaker_b_index,
        }

    def _build_faiss_index(self, texts: list[str]) -> faiss.Index | None:
        if not texts:
            return None
        embeddings = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        index = faiss.IndexFlatIP(self.embedding_dim)  # inner product on L2-normalised = cosine sim
        index.add(np.array(embeddings, dtype=np.float32))
        return index

    # ------------------------------------------------------------------
    # Step 3: Retrieve top-k memories for a query
    # ------------------------------------------------------------------
    def retrieve(self, query: str, memories: list[str], index: faiss.Index | None, top_k: int | None = None) -> list[str]:
        if index is None or not memories:
            return []
        k = min(top_k or self.top_k, len(memories))
        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(query_emb, k)
        return [memories[i] for i in indices[0] if i < len(memories)]

    # ------------------------------------------------------------------
    # Step 4: Generate answer via LLM
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Full pipeline: ingest → retrieve → answer for one conversation
    # ------------------------------------------------------------------
    def process_conversation(self, item: dict, conv_idx: int) -> list[dict]:
        conversation = item["conversation"]
        qa_pairs = item["qa"]

        mem = self.build_memories(conversation)
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

        return results


def main():
    parser = argparse.ArgumentParser(description="Run naive memory baseline on LoCoMo")
    parser.add_argument("--dataset", type=str, default="dataset/locomo10.json")
    parser.add_argument("--output", type=str, default="results/naive_baseline_results.json")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}")
    with open(args.dataset) as f:
        data = json.load(f)

    baseline = NaiveMemoryBaseline(
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
    )

    # Resume: load existing results if the output file already exists
    all_results = defaultdict(list)
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        for k, v in existing.items():
            all_results[int(k)] = v
        done = sum(len(v) for v in all_results.values())
        print(f"Resuming — found {len(all_results)} conversations ({done} questions) already done.")

    total_questions = sum(len(v) for v in all_results.values())

    for idx, item in enumerate(tqdm(data, desc="Processing conversations")):
        if idx in all_results and len(all_results[idx]) > 0:
            print(f"  Conv {idx}: already done ({len(all_results[idx])} questions), skipping")
            continue

        results = baseline.process_conversation(item, idx)
        all_results[idx] = results
        total_questions += len(results)

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  Conv {idx}: {len(results)} questions answered")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone! {total_questions} questions answered across {len(data)} conversations.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
