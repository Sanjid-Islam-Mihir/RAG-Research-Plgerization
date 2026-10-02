"""
Step 6: LLM
-------------
Loads microsoft/Phi-4-mini-instruct locally via `transformers` and
wraps it as a LangChain-compatible chat model.

No Hugging Face API key needed: this repo is public/ungated, so
`from_pretrained` downloads the weights directly. The first run pulls
the model (a few GB) into your local Hugging Face cache
(~/.cache/huggingface); every run after that is fully offline.
"""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

DEFAULT_MODEL_ID = "microsoft/Phi-4-mini-instruct"


def get_llm(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
) -> ChatHuggingFace:
    """
    Loads the model + tokenizer and returns a LangChain chat model.

    Args:
        model_id: Hugging Face repo id. Swap this to try a different
            free model later (e.g. "Qwen/Qwen2.5-7B-Instruct") without
            changing any other file.
        max_new_tokens: generation length cap — Q&A output is short, so
            1024 is generous headroom.
        temperature: low (0.2-0.4) keeps answers grounded in the source
            text rather than creative; raise it if outputs feel too
            repetitive across runs.

    Returns:
        A ChatHuggingFace instance usable anywhere LangChain expects a
        chat model, e.g. in an LCEL chain: `prompt | llm | parser`.

    Note: device_map="auto" uses a GPU automatically if one is
    available and visible to PyTorch; otherwise it falls back to CPU
    (Phi-4-mini is small enough — 3.8B params — to run on CPU, just
    more slowly).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
    )

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return ChatHuggingFace(llm=llm)


if __name__ == "__main__":
    chat_model = get_llm()
    response = chat_model.invoke("Say hello in one sentence.")
    print(response.content)
