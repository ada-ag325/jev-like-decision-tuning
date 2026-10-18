"""Sentiment prompts and the mapping from class IDs to answer positions."""

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

LABEL_NAMES = ("negative", "neutral", "positive")
LETTERS = ("A", "B", "C")
DEFAULT_ORDER = (0, 1, 2)
INSTRUCTION = "Classify the sentiment of the following text."


def _validate_order(order: Sequence[int]) -> tuple[int, ...]:
    order = tuple(order)
    if (
        any(not isinstance(item, Integral) or isinstance(item, bool) for item in order)
        or sorted(order) != list(DEFAULT_ORDER)
    ):
        raise ValueError("order must be a permutation of (0, 1, 2)")
    return order


def build_prompt(
    text: str,
    order: Sequence[int] = DEFAULT_ORDER,
    *,
    compact_options: bool = False,
    include_reply_instruction: bool = True,
) -> str:
    """Build a user prompt; order[position] is the class shown at that position.

    compact_options preserves the original 100-example baseline's spacing.
    Training and full evaluation use the default spaced option format.
    """
    order = _validate_order(order)
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    separator = ":" if compact_options else ": "
    options = "\n".join(
        f"{LETTERS[position]}{separator}{LABEL_NAMES[label_id]}"
        for position, label_id in enumerate(order)
    )
    prompt = f"{INSTRUCTION}\n\nText: {text}\n\n{options}\n\nAnswer:"
    if include_reply_instruction:
        prompt += "\nReply with only A, B, or C."
    return prompt


def format_example(
    example: Mapping[str, Any],
    order: Sequence[int] = DEFAULT_ORDER,
    *,
    compact_options: bool = False,
    include_reply_instruction: bool = True,
) -> dict[str, Any]:
    """Return a prompt, a zero-based answer position, and its answer letter.

    The target is an option position, not necessarily the original class ID.
    """
    order = _validate_order(order)
    label = example["label"]
    if not isinstance(label, Integral) or isinstance(label, bool) or label not in DEFAULT_ORDER:
        raise ValueError("label must be an integer class ID: 0, 1, or 2")
    target = order.index(label)
    return {
        "prompt": build_prompt(
            example["text"],
            order,
            compact_options=compact_options,
            include_reply_instruction=include_reply_instruction,
        ),
        "target": target,
        "target_letter": LETTERS[target],
    }


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Apply the notebook's chat template without adding the target answer."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
