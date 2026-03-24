from __future__ import annotations

from examples.mas_orchestra.parsing import extract_unique_boxed_letter
from examples.mas_orchestra.types import ReasoningSample


def gold_letter(answer_index: int) -> str:
    return chr(ord("A") + answer_index)


def compute_mca(sample: ReasoningSample, final_answer_text: str) -> float:
    predicted_letter, _ = extract_unique_boxed_letter(final_answer_text)
    return 1.0 if predicted_letter and predicted_letter.upper() == gold_letter(sample.answer_index) else 0.0
