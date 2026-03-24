from __future__ import annotations

from examples.mas_orchestra.pricing import ModelPricing
from examples.mas_orchestra.types import AttemptRecord, ReasoningSample


def build_model_pricing_table(sub_models: list[str]) -> str:
    lines = ["| Model | Input $/1K | Output $/1K |", "|---|---:|---:|"]
    for model_name in sub_models:
        pricing = ModelPricing.resolve_pricing(model_name)
        if pricing is None:
            lines.append(f"| {model_name} | N/A | N/A |")
        else:
            lines.append(f"| {model_name} | ${pricing['input']:.5f} | ${pricing['output']:.5f} |")
    return "\n".join(lines)


def build_main_prompt(
    sample: ReasoningSample,
    attempt_history_text: str,
    attempt_index: int,
    max_attempts: int,
    sub_models: list[str],
    threshold: float,
) -> str:
    remaining = max_attempts - attempt_index + 1
    pricing_table = build_model_pricing_table(sub_models)
    return f"""
You are the MainAgent (Orchestrator) for multimodal scientific reasoning.

You MUST choose one action each turn:
1) delegate_task
2) submit

DECISION PROCESS:
1. Review attempt history and identify what remains unresolved.
2. If the best boxed answer is valid and confidence >= {threshold:.2f}, prefer submit.
3. Otherwise delegate only the unresolved part.

BUDGET AWARENESS:
- Attempts are limited, so each delegation has a cost.
- Choose cheaper models for simple checks.
- Choose stronger models for complex or final checks.

MODEL PRICING:
{pricing_table}

Task:
Question: {sample.question}
Options: {sample.options}

Attempt history:
{attempt_history_text}

Progress:
Attempt {attempt_index}/{max_attempts}, remaining={remaining}
Available sub-models: {sub_models}

Output JSON only:
{{
  "action": "delegate_task|submit",
  "reasoning": "short rationale tied to remaining uncertainty",
  "model": "one of available sub-models (required if delegate_task)",
  "instruction": "specific reflection instruction (required if delegate_task)",
  "submit_reason": "why the evidence is enough to finalize (required if submit)"
}}
""".strip()


def _format_attempts(prior_attempts: list[AttemptRecord]) -> str:
    if not prior_attempts:
        return "No prior attempts."

    lines: list[str] = []
    for attempt in prior_attempts:
        delegate = attempt.delegate_result
        lines.append(
            (
                f"Attempt {attempt.attempt_index}: boxed={delegate.boxed_letter}, "
                f"confidence={delegate.confidence}, parse_ok={delegate.parse_ok}, "
                f"error={delegate.error or '-'}"
            )
        )
        if delegate.reasoning_summary:
            lines.append(f"Summary: {delegate.reasoning_summary}")
    return "\n".join(lines)


def build_sub_prompt(
    sample: ReasoningSample,
    prior_attempts: list[AttemptRecord],
    instruction: str,
) -> str:
    reflection_instruction = instruction.strip() if instruction and instruction.strip() else "Re-check key evidence before final answer."
    options = "\n".join(f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(sample.options))
    return f"""
You are a specialized SubAgent for scientific reasoning.

Follow this instruction from MainAgent:
{reflection_instruction}

Question:
{sample.question}

Options:
{options}

Prior attempts:
{_format_attempts(prior_attempts)}

Output JSON only:
{{
  "reasoning": "step-by-step reasoning",
  "final_answer": "\\boxed{{A}}",
  "confidence": 0.00
}}
""".strip()
