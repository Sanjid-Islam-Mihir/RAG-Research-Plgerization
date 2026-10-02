"""
Answer sheet I/O
-------------------
Splits generated Q&A into two separate files so the reference answer
is never sitting in the same file the author writes their answer
into - otherwise there'd be nothing stopping someone from just
reading the correct answer off the page before "answering" it.

- qa_output.json    -> id + question + reference answer. Kept by the
                       reviewer only; the author never sees this file.
- answer_sheet.json -> id + question + a blank candidate_answer field.
                       This is what the author fills in - by hand for
                       now, or later by a voice-to-text script writing
                       its transcription into that same field.
"""

import json
from typing import Dict, List

QA_OUTPUT_PATH = "qa_output.json"
ANSWER_SHEET_PATH = "answer_sheet.json"


def save_qa_and_answer_sheet(
    qa_pairs: List[Dict],
    qa_output_path: str = QA_OUTPUT_PATH,
    answer_sheet_path: str = ANSWER_SHEET_PATH,
) -> None:
    """Assigns a stable id to each pair, then writes both files."""
    for i, pair in enumerate(qa_pairs, start=1):
        pair["id"] = i

    with open(qa_output_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    answer_sheet = [
        {"id": pair["id"], "question": pair["question"], "candidate_answer": ""}
        for pair in qa_pairs
    ]
    with open(answer_sheet_path, "w") as f:
        json.dump(answer_sheet, f, indent=2)


def load_and_merge(
    qa_output_path: str = QA_OUTPUT_PATH,
    answer_sheet_path: str = ANSWER_SHEET_PATH,
) -> List[Dict]:
    """
    Reads both files back and merges them by id into the shape
    evaluate_answers() expects: question, answer (reference), and
    candidate_answer.

    Raises ValueError if any candidate_answer is still blank, since
    that means the answer sheet hasn't been fully filled in yet.
    """
    with open(qa_output_path) as f:
        reference_by_id = {p["id"]: p for p in json.load(f)}

    with open(answer_sheet_path) as f:
        answer_sheet = json.load(f)

    merged = []
    unanswered = []
    for entry in answer_sheet:
        ref = reference_by_id[entry["id"]]
        candidate_answer = entry.get("candidate_answer", "").strip()
        if not candidate_answer:
            unanswered.append(entry["id"])
        merged.append(
            {
                "id": entry["id"],
                "question": ref["question"],
                "answer": ref["answer"],
                "candidate_answer": candidate_answer,
            }
        )

    if unanswered:
        raise ValueError(
            f"Question id(s) {unanswered} still have an empty 'candidate_answer' "
            f"in {answer_sheet_path}. Fill those in before evaluating."
        )

    return merged
