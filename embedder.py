from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    embedder = get_embedder()
    vector = embedder.embed_query("This paper proposes a novel attention mechanism.")
    print(f"Embedding dimension: {len(vector)}")
