"""
Entry point for the generation half of the project.

Give it a PDF and it runs: parse -> chunk -> embed -> store ->
retrieve -> generate reference Q&A. It writes two files:
  - qa_output.json    (question + reference answer - for the reviewer)
  - answer_sheet.json (question + a blank space to write the answer)

Fill in answer_sheet.json - by hand, or later by wiring up a
voice-to-text script that writes into its "candidate_answer" field -
then run evaluate.py to score it. This is split into two runs on
purpose: the answer sheet needs a real gap in time to be filled in,
so it can't safely be folded into one uninterrupted script.

Usage: python main.py path/to/paper.pdf
"""

import sys

import yaml

from pdf_parser import parse_pdf
from chunker import chunk_documents
from embedder import get_embedder
from vector_store import build_vector_store
from retriever import get_diverse_chunks, DEFAULT_QUERY
from llm import get_llm
from qa_generator import generate_qa_pairs
from answer_sheet import save_qa_and_answer_sheet, QA_OUTPUT_PATH, ANSWER_SHEET_PATH


def run(pdf_path: str, num_questions: int = 5, retrieval_query: str = DEFAULT_QUERY):
    print(f"[1/7] Parsing PDF: {pdf_path}")
    documents = parse_pdf(pdf_path)

    print("[2/7] Chunking")
    chunks = chunk_documents(documents)

    print("[3/7] Loading embedding model")
    embedder = get_embedder()

    print("[4/7] Embedding + storing in Chroma")
    vector_store = build_vector_store(chunks, embedder)

    print("[5/7] Retrieving diverse chunks for coverage")
    retrieved = get_diverse_chunks(vector_store, retrieval_query, k=num_questions + 1)

    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    llm_params = params["llm"]

    print(f"[6/7] Loading LLM ({llm_params['model_id']})")
    llm = get_llm(
        model_id=llm_params["model_id"],
        max_new_tokens=llm_params["max_new_tokens"],
        temperature=llm_params["temperature"],
    )

    print("[7/7] Generating reference Q&A from the paper")
    qa_pairs = generate_qa_pairs(retrieved, llm, num_questions=num_questions)

    save_qa_and_answer_sheet(qa_pairs)
    print(f"      -> reference answers saved to {QA_OUTPUT_PATH}")
    print(f"      -> blank answer sheet saved to {ANSWER_SHEET_PATH}")

    print(
        f"\nNext: fill in 'candidate_answer' for each question in {ANSWER_SHEET_PATH} "
        f"(by hand, or via your voice-to-text step), then run:\n"
        f"    python evaluate.py"
    )

    return qa_pairs


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_paper.pdf"
    run(pdf_path)
