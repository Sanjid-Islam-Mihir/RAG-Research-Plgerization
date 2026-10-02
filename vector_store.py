from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "research_papers"


def build_vector_store(
    chunks: List[Document],
    embedder: HuggingFaceEmbeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return vector_store


def load_vector_store(
    embedder: HuggingFaceEmbeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Loads an already-persisted Chroma DB without re-embedding anything."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_directory,
    )


if __name__ == "__main__":
    from pdf_parser import parse_pdf
    from chunker import chunk_documents
    from embedder import get_embedder

    docs = parse_pdf("sample_paper.pdf")
    chunks = chunk_documents(docs)
    embedder = get_embedder()
    store = build_vector_store(chunks, embedder)
    print(f"Stored {len(chunks)} chunks in Chroma at '{DEFAULT_PERSIST_DIR}'")
