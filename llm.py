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

import torch
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

    Note: "auto" device mapping is only used when a GPU is actually
    available. On a CPU-only machine, "auto" can decide there isn't
    enough RAM and silently spread the model across disk instead
    (you'll see a warning like "parameters are on the meta device
    because they were offloaded to the cpu and disk") — that path is
    unreliable and often fails or hangs partway through generation.
    Loading straight onto CPU with no device_map avoids it entirely.

    Loaded in bfloat16 rather than float32: this is a 3.8B-parameter
    model, so float32 (4 bytes/param) needs ~15GB for weights alone —
    on a 16GB machine that leaves almost nothing for the OS, Python,
    or the embedding model, and often gets silently killed by the OS
    rather than raising a catchable Python error. bfloat16 (2
    bytes/param) cuts that to ~7.6GB. low_cpu_mem_usage=True also
    avoids a memory spike during the loading step itself.

    If it still runs out of memory on this machine, that means even
    bfloat16 doesn't fit, and switching to a quantized runtime (e.g.
    Ollama) is the real fix — not something further tuning here can
    solve.
    """
    has_gpu = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if has_gpu else None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if not has_gpu:
        model = model.to("cpu")

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
