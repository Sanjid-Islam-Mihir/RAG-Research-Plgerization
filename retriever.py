from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List

DEFAULT_QUERY = "key contributions, methodology, and results of this paper"


def get_diverse_chunks(
    vector_store: Chroma,
    query: str = DEFAULT_QUERY,
    k: int = 6,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> List[Document]:

    return vector_store.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )


def get_retriever(vector_store: Chroma, k: int = 6):
    """
    Standard LangChain retriever object (LCEL-compatible), configured
    with the same MMR strategy, in case you want to build a chain
    with `| retriever |` instead of calling get_diverse_chunks directly.
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5},
    )


if __name__ == "__main__":
    from embedder import get_embedder
    from vector_store import load_vector_store

    embedder = get_embedder()
    store = load_vector_store(embedder)
    chunks = get_diverse_chunks(store)
    print(f"Retrieved {len(chunks)} diverse chunks")
    for c in chunks:
        print(f"- page {c.metadata.get('page')}, chunk {c.metadata.get('chunk_id')}")
