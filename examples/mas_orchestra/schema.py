from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningSample:
    task_id: str
    question: str
    options: list[str]
    answer_index: int
    steps: list[str]
    discipline: str
    images: list[Any]


@dataclass
class DelegateRequest:
    task_id: str
    question: str
    options: list[str]
    images: list[Any]
    prior_attempts: list["AttemptRecord"]
    model: str
    instruction: str


@dataclass
class DelegateResult:
    raw_answer_text: str
    boxed_letter: str | None
    confidence: float | None
    reasoning_summary: str
    parse_ok: bool
    error: str | None
    cost: float
    input_tokens: int = 0
    output_tokens: int = 0
    provider_model: str | None = None
    response_id: str | None = None
    running_total_cost: float | None = None
    cost_budget: float | None = None
    budget_exceeded: bool = False


@dataclass
class SubmitResult:
    final_answer_text: str
    final_boxed_letter: str | None
    done: bool
    reason: str
    attempt_count: int


@dataclass
class MainAction:
    action: str
    reasoning: str
    task_type: str | None = None
    difficulty: str | None = None
    model: str | None = None
    instruction: str | None = None
    submit_reason: str | None = None


@dataclass
class AttemptRecord:
    attempt_index: int
    model: str
    instruction: str
    delegate_result: DelegateResult
    main_reasoning: str = ""


@dataclass
class WorkflowMetrics:
    mca: float
    attempt_count: int
    main_policy_calls: int
    delegate_policy_calls: int
    external_delegate_calls: int
    models_used: list[str] = field(default_factory=list)
    model_usage: dict[str, int] = field(default_factory=dict)
