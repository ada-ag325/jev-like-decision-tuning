from itertools import permutations

import pytest

from jev_decision_tuning.prompts import build_prompt, format_example


@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("label", range(3))
def test_target_points_to_the_correct_class_in_every_permutation(order, label):
    result = format_example({"text": "Example", "label": label}, order)
    answer_line = result["prompt"].split("\n\n")[2].splitlines()[result["target"]]
    expected_class = ("negative", "neutral", "positive")[label]
    assert answer_line == f'{result["target_letter"]}: {expected_class}'


def test_original_prompt_variants_are_preserved_exactly():
    prefix = "Classify the sentiment of the following text.\n\nText: Great!\n\n"
    suffix = "\n\nAnswer:\nReply with only A, B, or C."
    assert build_prompt("Great!") == prefix + "A: negative\nB: neutral\nC: positive" + suffix
    assert build_prompt("Great!", compact_options=True) == (
        prefix + "A:negative\nB:neutral\nC:positive" + suffix
    )
    display = format_example(
        {"text": "Great!", "label": 2},
        compact_options=True,
        include_reply_instruction=False,
    )
    assert display == {
        "prompt": prefix + "A:negative\nB:neutral\nC:positive\n\nAnswer:",
        "target": 2,
        "target_letter": "C",
    }


@pytest.mark.parametrize("order", [(0, 0, 2), (0, 1), (0, 1, 3), (False, 1, 2)])
def test_invalid_orders_are_rejected(order):
    with pytest.raises(ValueError, match="permutation"):
        build_prompt("Example", order)


@pytest.mark.parametrize("label", [-1, 3, "positive", True])
def test_invalid_labels_cannot_silently_select_an_answer(label):
    with pytest.raises(ValueError, match="class ID"):
        format_example({"text": "Example", "label": label})
