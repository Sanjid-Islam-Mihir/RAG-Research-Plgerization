from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from typing import List, Literal

SplitterType = Literal["recursive", "token"]


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    splitter_type: SplitterType = "recursive",
) -> List[Document]:
    """
    Split documents into overlapping chunks.

    Args:
        documents: output of parse_pdf().
        chunk_size: max size of each chunk (chars for 'recursive', tokens for 'token').
        chunk_overlap: overlap between consecutive chunks, keeps context continuity.
        splitter_type:
            'recursive' - character-based, tries to break on paragraph/sentence
                          boundaries first. Good general default.
            'token'     - splits by token count, useful when you need to respect
                          an LLM's context window precisely.

    Returns:
        List of chunked Document objects. Original metadata (source, page) is
        preserved, and a 'chunk_id' is added to each for traceability.
    """
    if splitter_type == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif splitter_type == "token":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"Unknown splitter_type: {splitter_type}")

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


if __name__ == "__main__":
    from pdf_parser import parse_pdf

    docs = parse_pdf("sample_paper.pdf")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages")
    print("--- Preview of chunk 0 ---")
    print(chunks[0].page_content[:300])
