import sys
from pdf_parser import parse_pdf
from chunker import chunk_documents
from embedder import get_embedder
from vector_store import build_vector_store


def run_pipeline(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    splitter_type: str = "recursive",
    embedding_model: str = None,
    persist_directory: str = "chroma_db",
    collection_name: str = "research_papers",
):
    print(f"[1/4] Parsing PDF: {pdf_path}")
    documents = parse_pdf(pdf_path)
    print(f"      -> {len(documents)} pages")

    print(f"[2/4] Chunking (size={chunk_size}, overlap={chunk_overlap}, type={splitter_type})")
    chunks = chunk_documents(documents, chunk_size, chunk_overlap, splitter_type)
    print(f"      -> {len(chunks)} chunks")

    print("[3/4] Loading embedding model")
    embedder = get_embedder(embedding_model) if embedding_model else get_embedder()

    print(f"[4/4] Embedding + storing in Chroma ('{persist_directory}')")
    vector_store = build_vector_store(chunks, embedder, persist_directory, collection_name)
    print("Done. Vector store ready for retrieval.")

    return vector_store


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_paper.pdf"
    run_pipeline(pdf_path)
