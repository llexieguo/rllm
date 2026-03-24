from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

from examples.mas_orchestra.parsing import extract_confidence, extract_unique_boxed_letter, parse_json_fragment
from examples.mas_orchestra.prompts import build_sub_prompt
from examples.mas_orchestra.types import DelegateRequest, DelegateResult, ReasoningSample


def build_delegate_result(raw_text: str, *, cost: float = 0.0, input_tokens: int = 0, output_tokens: int = 0) -> DelegateResult:
    parsed = parse_json_fragment(raw_text)
    confidence = extract_confidence(raw_text, parsed)

    answer_text = raw_text
    if parsed and isinstance(parsed.get("final_answer"), str):
        parsed_answer = parsed["final_answer"]
        parsed_letter, _ = extract_unique_boxed_letter(parsed_answer)
        answer_text = parsed_answer if parsed_letter is not None else raw_text

    boxed_letter, boxed_error = extract_unique_boxed_letter(answer_text)
    reasoning = ""
    if parsed and isinstance(parsed.get("reasoning"), str):
        reasoning = parsed["reasoning"].strip()
    if not reasoning:
        reasoning = raw_text.strip()

    parse_ok = boxed_error is None and confidence is not None
    error = boxed_error
    if confidence is None:
        parse_ok = False
        error = f"{error}; missing confidence" if error else "Missing confidence"

    return DelegateResult(
        raw_answer_text=raw_text,
        boxed_letter=boxed_letter,
        confidence=confidence,
        reasoning_summary=reasoning,
        parse_ok=parse_ok,
        error=error,
        cost=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class SubModelClient(ABC):
    @abstractmethod
    async def run(self, request: DelegateRequest) -> DelegateResult:
        raise NotImplementedError


class MockSubModelClient(SubModelClient):
    def __init__(self, responses: dict[str, list[str] | str] | list[str] | str | None = None):
        if isinstance(responses, dict):
            self.responses = {
                key: (value[:] if isinstance(value, list) else value)
                for key, value in responses.items()
            }
        elif isinstance(responses, list):
            self.responses = {"*": responses[:]}
        elif isinstance(responses, str):
            self.responses = {"*": responses}
        else:
            self.responses = {}
        self.call_log: list[dict[str, Any]] = []

    def _resolve_response(self, model_name: str, request: DelegateRequest) -> str:
        configured = self.responses.get(model_name, self.responses.get("*"))
        if isinstance(configured, list):
            if configured:
                return configured.pop(0)
        elif isinstance(configured, str):
            return configured

        default_letter = "A" if request.options else "A"
        return (
            '{"reasoning":"mock external delegate reasoning",'
            f'"final_answer":"\\\\boxed{{{default_letter}}}","confidence":0.61}}'
        )

    async def run(self, request: DelegateRequest) -> DelegateResult:
        self.call_log.append(
            {
                "task_id": request.task_id,
                "model": request.model,
                "instruction": request.instruction,
            }
        )
        raw_text = self._resolve_response(request.model, request)
        return build_delegate_result(raw_text)


def task_to_reasoning_sample(task: dict[str, Any]) -> ReasoningSample:
    return ReasoningSample(
        task_id=str(task.get("task_id", task.get("idx", task.get("id", "")))),
        question=str(task.get("question", "")),
        options=[str(option) for option in task.get("options", [])],
        answer_index=int(task.get("answer_index", task.get("answer", 0))),
        steps=[str(step) for step in task.get("steps", [])],
        discipline=str(task.get("discipline", "unknown")),
        images=list(task.get("images", []) or []),
    )


def build_delegate_request(sample: ReasoningSample, prior_attempts: list[Any], model: str, instruction: str) -> DelegateRequest:
    return DelegateRequest(
        task_id=sample.task_id,
        question=sample.question,
        options=sample.options,
        images=copy.copy(sample.images),
        prior_attempts=prior_attempts,
        model=model,
        instruction=instruction,
    )
