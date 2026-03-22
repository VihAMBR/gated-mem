"""
Quick smoke test: runs 5 questions from the first conversation to verify
the full pipeline (embed → retrieve → LLM answer → eval) works end-to-end.

Usage:
    python test_quick.py
"""

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

from run_naive_baseline import NaiveMemoryBaseline
from metrics.llm_judge import evaluate_llm_judge

DATASET = "dataset/locomo10.json"
NUM_QUESTIONS = 5

def main():
    if os.getenv("OPENAI_API_KEY", "").startswith("your-"):
        print("ERROR: Please set your real OPENAI_API_KEY in .env first!")
        print("  Edit .env and replace 'your-openai-api-key-here' with your actual key.")
        return

    with open(DATASET) as f:
        data = json.load(f)

    print("Loading embedding model...")
    baseline = NaiveMemoryBaseline(top_k=30)

    item = data[0]
    conversation = item["conversation"]
    qa_pairs = [q for q in item["qa"] if int(q["category"]) != 5][:NUM_QUESTIONS]

    print(f"\nBuilding memories for conversation 0 ({conversation['speaker_a']} & {conversation['speaker_b']})...")
    mem = baseline.build_memories(conversation)
    print(f"  {len(mem['speaker_a_memories'])} + {len(mem['speaker_b_memories'])} memories")

    correct = 0
    total = 0

    for qa in qa_pairs:
        question = qa["question"]
        gold = qa["answer"]
        category = qa["category"]

        a_ret = baseline.retrieve(question, mem["speaker_a_memories"], mem["speaker_a_index"])
        b_ret = baseline.retrieve(question, mem["speaker_b_memories"], mem["speaker_b_index"])

        response, resp_time = baseline.answer_question(
            question, mem["speaker_a_name"], mem["speaker_b_name"], a_ret, b_ret
        )

        judge_score = evaluate_llm_judge(question, gold, response)
        correct += judge_score
        total += 1

        status = "CORRECT" if judge_score else "WRONG"
        print(f"\n[{status}] Category {category}")
        print(f"  Q: {question}")
        print(f"  Gold: {gold}")
        print(f"  Pred: {response}")
        print(f"  LLM time: {resp_time:.1f}s")

    print(f"\n{'='*60}")
    print(f"Quick test: {correct}/{total} correct ({100*correct/total:.0f}%)")
    print("Pipeline is working! You can now run the full benchmark:")
    print("  python run_naive_baseline.py")


if __name__ == "__main__":
    main()
