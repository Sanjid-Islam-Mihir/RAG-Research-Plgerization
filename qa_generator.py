"""
Step 7: Question & Answer Generation
---------------------------------------
Takes the retrieved chunks + the LLM and produces structured Q&A
pairs. PROMPT_TEMPLATE below is a starting point, not a final answer —
this is the part you said you want to write/tune yourselves, so treat
it as a draft: adjust the instructions, add few-shot examples, change
the question mix, etc.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Starter prompt: this is the part meant to be rewritten/tuned ---
PROMPT_TEMPLATE = """You are helping a reviewer verify that a research paper was genuinely \
written by its claimed author(s), by preparing questions that only someone who deeply \
understands the paper's actual content could answer correctly.

Using ONLY the excerpts below, write {num_questions} question-and-answer pairs.

Rules:
- Questions must be specific to THIS paper's content (its methods, results, design \
choices), not generic questions about the general topic.
- Include a mix of conceptual questions ("why did they choose X approach over Y") and \
factual ones ("what dataset/metric/value did they report").
- Answers must be fully supported by the excerpts below — do not use outside knowledge.
- Return ONLY a JSON array, with no extra text before or after it, in this exact shape:
[{{"question": "...", "answer": "..."}}, ...]

Excerpts from the paper:
{context}
"""


def format_context(chunks: List[Document]) -> str:
    """Joins retrieved chunks into one context block, tagging each with its page."""
    parts = []
    for chunk in chunks:
        page = chunk.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_qa_pairs(chunks: List[Document], llm, num_questions: int = 5) -> list:
    """
    Runs PROMPT_TEMPLATE against the LLM and parses the response into
    a Python list of {"question": ..., "answer": ...} dicts.

    Args:
        chunks: diverse chunks from get_diverse_chunks().
        llm: a chat model from get_llm().
        num_questions: how many Q&A pairs to request.

    Returns:
        List of dicts, each with "question" and "answer" keys.

    Note: JsonOutputParser expects the model to return valid JSON. Small
    local models occasionally wrap it in extra text despite instructions —
    if you hit parse errors, either lower temperature in get_llm(), or
    swap JsonOutputParser for a more lenient parser / add a retry.
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    context = format_context(chunks)
    return chain.invoke({"context": context, "num_questions": num_questions})


if __name__ == "__main__":
    from embedder import get_embedder
    from vector_store import load_vector_store
    from retriever import get_diverse_chunks
    from llm import get_llm

    embedder = get_embedder()
    store = load_vector_store(embedder)
    chunks = get_diverse_chunks(store, k=5)

    llm = get_llm()
    qa_pairs = generate_qa_pairs(chunks, llm, num_questions=5)

    for i, pair in enumerate(qa_pairs, 1):
        print(f"Q{i}: {pair['question']}")
        print(f"A{i}: {pair['answer']}\n")
