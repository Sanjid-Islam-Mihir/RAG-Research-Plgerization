# RAG Paper QA

A LangChain pipeline that reads a research paper, generates viva-style questions with
reference answers drawn only from the paper's own content, and scores a candidate's
live answers against those references — a decision-support signal for verifying
someone genuinely understands/wrote a paper, not a text-plagiarism checker.

## Project layout

```
rag_paper_qa/
├── config.yaml           # all tunable parameters
├── requirements.txt
├── data/                 # put your research paper PDF(s) here
└── src/
    ├── pdf_parser.py     # Step 1 - extract text from the PDF
    ├── chunker.py        # Step 2 - split text into chunks
    ├── embedder.py       # Step 3 - load the embedding model
    ├── vector_store.py   # Step 4 - store/load chunks in Chroma
    ├── retriever.py      # Step 5 - pull diverse chunks back out
    ├── llm.py            # Step 6 - load microsoft/Phi-4-mini-instruct
    ├── qa_generator.py   # Step 7 - generation prompt + chain
    ├── evaluator.py      # Step 8 - scoring logic (similarity + LLM judge)
    ├── answer_sheet.py   # splits reference answers from the blank answer sheet
    ├── main.py           # entry point: PDF -> qa_output.json + answer_sheet.json
    └── evaluate.py       # entry point: answer_sheet.json -> evaluation_report.json
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

No Hugging Face API key is required — both the embedding model and
Phi-4-mini-instruct are public/ungated and run entirely locally.

## Usage

1. Drop your paper into `data/`, e.g. `data/my_paper.pdf`.
2. From inside `src/`, generate the reference Q&A:
   ```bash
   python main.py ../data/my_paper.pdf
   ```
   This produces two files in `src/`:
   - `qa_output.json` — questions + reference answers, for the reviewer only
   - `answer_sheet.json` — same questions, blank `candidate_answer` fields
3. Fill in `answer_sheet.json`'s `candidate_answer` fields — by hand for now, or
   later via a voice-to-text step that writes into the same field.
4. Score it:
   ```bash
   python evaluate.py
   ```
   This produces `evaluation_report.json`: a per-question score plus an overall
   summary verdict for the reviewer.

## Notes

- `config.yaml` centralizes every tunable value (chunk size, retrieval k, LLM
  temperature, evaluation thresholds) and is meant to become `params.yaml` once
  DVC is added for chunking experiments.
- The prompts in `qa_generator.py` (`PROMPT_TEMPLATE`) and `evaluator.py`
  (`JUDGE_PROMPT_TEMPLATE`) are starting points, meant to be tuned.
- Swap the LLM by changing `model_id` in `llm.py` / `config.yaml` — nothing else
  needs to change.
