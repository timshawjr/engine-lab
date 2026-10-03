"""Grounded answer generation with Qwen3-1.7B, plus the refusal detector.

TWO MEASURED FAILURES SHAPE THIS MODULE.

1. Qwen3 is a reasoning model. Left to its own chat template it spends the
   entire token budget emitting ``<think>...</think>`` and returns no answer at
   all -- every measured answer was 100% reasoning text. The fix is to disable
   the template and prefill the assistant turn with an EMPTY think block, which
   closes reasoning immediately and makes the budget buy an answer.

2. The passages must be passed whole. Truncating each retrieved chunk to 110
   words cut off the very sentence being asked about, and both the 0.6B and
   1.7B models then answered "NOT IN PROVIDED PAGES" for questions whose answers
   had in fact been retrieved correctly. A wrong abstention reads as a correct
   refusal, so truncation here is worse than it looks.

Decoding is greedy so a booth visitor asking the same question twice sees the
same answer.
"""

from __future__ import annotations

from app.rag.retrieval import Passage

# Qwen3 ChatML control tokens. Written explicitly rather than relying on the
# tokenizer, because apply_chat_template is disabled (see module docstring).
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def build_prompt(system: str, passages: list[Passage], question: str) -> str:
    """Build the ChatML prompt, with the assistant turn prefilled to skip reasoning.

    Args:
        system: Instruction telling the model to answer only from the passages
            and to emit the abstention marker when they do not support one.
        passages: Retrieved chunks, already ranked. Their text is passed in
            full -- never truncated.
        question: The visitor's question.
    """
    context = "\n\n".join(
        f"[p.{passage.page}] {' '.join(passage.text.split())}" for passage in passages
    )
    user = f"Passages:\n{context}\n\nQuestion: {question}" if context else f"Question: {question}"
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{user}{IM_END}\n"
        f"{IM_START}assistant\n{EMPTY_THINK}"
    )


def build_generation_config(max_new_tokens: int) -> "object":
    """Greedy, template-free generation config."""
    import openvino_genai as genai

    config = genai.GenerationConfig()
    config.do_sample = False
    config.temperature = 0.0
    config.max_new_tokens = int(max_new_tokens)
    # Applying the chat template re-enables Qwen3 reasoning mode, which
    # consumes the entire budget and yields no answer.
    config.apply_chat_template = False
    return config


def is_abstention(answer: str, marker: str) -> bool:
    """Whether the model declined rather than answered.

    An empty or whitespace-only answer is NOT an abstention: that is a failure
    to generate, and reporting it as a refusal would tell a visitor the
    passages did not support an answer when the model produced nothing at all.
    """
    text = (answer or "").strip()
    if not text:
        return False
    return marker.strip().lower() in text.lower()


def generate(pipeline: object, prompt: str) -> str:
    """Run one generation and return the answer text.

    Measured end-to-end on this machine: mean 2.1 s, max 3.4 s per question,
    which is why the worker runs this off the UI thread.
    """
    return str(pipeline.generate(prompt))