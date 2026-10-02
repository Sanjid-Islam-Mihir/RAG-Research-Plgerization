import json
import sys

from pdf_parser import parse_pdf
from chunker import chunk_documents
from embedder import get_embedder
from vector_store import build_vector_store
from retriever import get_diverse_chunks, DEFAULT_QUERY
from llm import get_llm
from qa_generator import generate_qa_pairs


def run_full_pipeline(
    pdf_path: str,
    num_questions: int = 5,
    retrieval_query: str = DEFAULT_QUERY,
):
    print(f"[1/6] Parsing PDF: {pdf_path}")
    documents = parse_pdf(pdf_path)

    print("[2/6] Chunking")
    chunks = chunk_documents(documents)

    print("[3/6] Loading embedding model")
    embedder = get_embedder()

    print("[4/6] Embedding + storing in Chroma")
    vector_store = build_vector_store(chunks, embedder)

    print("[5/6] Retrieving diverse chunks for coverage")
    retrieved = get_diverse_chunks(vector_store, retrieval_query, k=num_questions + 1)

    print("[6/6] Loading LLM and generating Q&A")
    llm = get_llm()
    qa_pairs = generate_qa_pairs(retrieved, llm, num_questions=num_questions)

    return qa_pairs


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_paper.pdf"
    qa_pairs = run_full_pipeline(pdf_path)

    print("\n=== Generated Q&A ===")
    for i, pair in enumerate(qa_pairs, 1):
        print(f"\nQ{i}: {pair['question']}")
        print(f"A{i}: {pair['answer']}")

    with open("qa_output.json", "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print("\nSaved to qa_output.json")
