from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List


def parse_pdf(pdf_path: str) -> List[Document]:

    resolved_path = str(Path(pdf_path).resolve())
    loader = PyPDFLoader(resolved_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No text could be extracted from: {resolved_path}")

    return documents


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_paper.pdf"
    docs = parse_pdf(path)
    print(f"Parsed {len(docs)} pages from {path}")
    print("--- Preview of page 1 ---")
    print(docs[0].page_content[:500])
