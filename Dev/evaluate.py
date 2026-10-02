"""
Entry point for the evaluation half of the project.

Run this only once answer_sheet.json has actually been filled in -
by hand, or by a voice-to-text script writing into its
"candidate_answer" fields. It merges the answer sheet back against
the reference answers, scores each pair, and saves a report.

Usage: python evaluate.py [qa_output.json] [answer_sheet.json]
"""

import json
import sys

from answer_sheet import ANSWER_SHEET_PATH, QA_OUTPUT_PATH, load_and_merge
from embedder import get_embedder
from evaluator import evaluate_answers, summarize
from llm import get_llm

if __name__ == "__main__":
    qa_output_path = sys.argv[1] if len(sys.argv) > 1 else QA_OUTPUT_PATH
    answer_sheet_path = sys.argv[2] if len(sys.argv) > 2 else ANSWER_SHEET_PATH

    try:
        qa_pairs = load_and_merge(qa_output_path, answer_sheet_path)
    except ValueError as e:
        print(f"Not ready to evaluate yet: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Missing file: {e}. Run main.py first to generate the Q&A and answer sheet.")
        sys.exit(1)

    print("Loading embedding model and LLM for scoring...")
    embedder = get_embedder()
    llm = get_llm()

    results = evaluate_answers(qa_pairs, embedder, llm)
    summary = summarize(results)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))

    with open("evaluation_report.json", "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
    print("\nFull report saved to evaluation_report.json")
