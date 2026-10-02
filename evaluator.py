"""
Step 8: Evaluation
---------------------
Compares the paper's real author's answers against the reference
answers the pipeline generated from the paper itself. A wide gap
between the two is the signal this project is built to catch: someone
who didn't really write/understand the paper usually can't reproduce
answers that match its actual content, even if the paper's text reads
fine on its own.

Two complementary checks, combined per question:
  1. Embedding similarity - fast, cheap, catches answers that are
     semantically off-topic or missing the key content entirely.
  2. LLM-as-judge - slower, but catches nuance a raw similarity score
     misses (e.g. an answer using different words that is factually
     wrong, or one that's vague/evasive).

This produces a decision-support signal for a human reviewer, not an
automated verdict: a low score means "look closer at this answer,"
not "this person is guilty."
"""

from typing import Dict, List

import numpy as np
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Starter prompt: tune this like the generation prompt ---
JUDGE_PROMPT_TEMPLATE = """You are checking whether a paper's author truly understands \
content the paper itself states, by comparing their live answer to a reference answer \
drawn directly from the paper.

Question: {question}

Reference answer (from the paper): {reference_answer}

Author's answer: {candidate_answer}

Score how well the author's answer matches the reference answer's meaning, 0-10:
- 9-10: matches closely, same key facts/reasoning, wording can differ.
- 5-8: partially matches, missing some detail or slightly imprecise.
- 0-4: contradicts, is vague/evasive, or misses the core point entirely.

Return ONLY JSON, no extra text, in this exact shape:
{{"score": <0-10 integer>, "verdict": "<one short phrase>", "reasoning": "<1-2 sentences>"}}
"""


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embedding_similarity_score(reference_answer: str, candidate_answer: str, embedder) -> float:
    """Returns a 0-1 semantic similarity score between the two answers."""
    ref_vec = embedder.embed_query(reference_answer)
    cand_vec = embedder.embed_query(candidate_answer)
    return cosine_similarity(ref_vec, cand_vec)


def llm_judge_score(question: str, reference_answer: str, candidate_answer: str, llm) -> Dict:
    """Returns {"score": int, "verdict": str, "reasoning": str} from the LLM."""
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT_TEMPLATE)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain.invoke(
        {
            "question": question,
            "reference_answer": reference_answer,
            "candidate_answer": candidate_answer,
        }
    )


def evaluate_answers(qa_pairs: List[Dict], embedder, llm) -> List[Dict]:
    """
    Args:
        qa_pairs: list of dicts, each with "question", "answer" (the
            reference answer from the paper) and "candidate_answer"
            (what the real author said live).
        embedder: from get_embedder().
        llm: from get_llm().

    Returns:
        The same list with "embedding_similarity" (float, 0-1) and
        "llm_judge" (dict: score/verdict/reasoning) added to each entry.
    """
    results = []
    for pair in qa_pairs:
        sim = embedding_similarity_score(pair["answer"], pair["candidate_answer"], embedder)
        judged = llm_judge_score(pair["question"], pair["answer"], pair["candidate_answer"], llm)
        results.append({**pair, "embedding_similarity": round(sim, 3), "llm_judge": judged})
    return results


def summarize(results: List[Dict]) -> Dict:
    """Aggregates per-question scores into one overall signal for the reviewer."""
    avg_sim = sum(r["embedding_similarity"] for r in results) / len(results)
    avg_llm = sum(r["llm_judge"]["score"] for r in results) / len(results)
    flagged = [r for r in results if r["llm_judge"]["score"] < 5 or r["embedding_similarity"] < 0.5]

    if avg_llm >= 7 and not flagged:
        verdict = "Answers are consistent with the paper's content."
    elif flagged:
        verdict = f"{len(flagged)} of {len(results)} answers didn't match well - needs a closer look."
    else:
        verdict = "Mixed match - review the lower-scoring answers below."

    return {
        "average_embedding_similarity": round(avg_sim, 3),
        "average_llm_score": round(avg_llm, 2),
        "flagged_count": len(flagged),
        "verdict": verdict,
    }
