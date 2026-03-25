from __future__ import annotations

from examples.mas_orchestra.pricing import ModelPricing
from examples.mas_orchestra.schema import AttemptRecord, ReasoningSample


def build_model_pricing_table(sub_models: list[str]) -> str:
    lines = ["| Model | Input $/1K | Output $/1K |", "|---|---:|---:|"]
    for model_name in sub_models:
        label = ModelPricing.format_model_label(model_name)
        pricing = ModelPricing.resolve_pricing(model_name)
        if pricing is None:
            lines.append(f"| {label} | N/A | N/A |")
        else:
            lines.append(f"| {label} | ${pricing['input']:.5f} | ${pricing['output']:.5f} |")
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
1. REVIEW attempt history and identify what failed vs. what was resolved.
2. EVALUATE answer validity:
   - If boxed answer is missing or invalid, continue with delegate_task.
   - If boxed answer is valid and confidence >= {threshold:.2f}, prefer submit.
3. DECIDE next action:
   - submit only when evidence is sufficient.
   - delegate_task only for the remaining uncertainty; do not repeat solved parts.

BUDGET AWARENESS:
- Attempts are limited, so each delegation has a cost.
- Choose cheaper models for simpler reasoning checks.
- Choose stronger models for complex reasoning or critical attempts.

MODEL PRICING (configured sub-models):
{pricing_table}

TASK TYPE IDENTIFICATION (MANDATORY):
Before selecting a model, classify the REMAINING task type.

Possible task types:

CALCULATION
- mathematical derivation
- equation solving
- symbolic or numeric computation
- unit conversion or formula application

SCI_REASONING
- multi-step scientific reasoning
- conceptual reasoning involving physics, chemistry, or biology
- hypothesis evaluation
- reasoning involving multiple scientific assumptions

VISION_REASONING
- interpreting diagrams, charts, or experimental figures
- extracting spatial or visual relationships
- multimodal reasoning combining text and image evidence

Focus only on the remaining uncertainty that still needs to be resolved.

DIFFICULTY ESTIMATION (MANDATORY):
After identifying the task type, classify the REMAINING work as EASY / MEDIUM / HARD.

EASY:
- straightforward computation with clear formula
- direct reasoning from already consistent evidence

MEDIUM:
- multi-step reasoning with moderate ambiguity
- calculation involving multiple intermediate steps
- visual interpretation with relatively clear evidence

HARD:
- ambiguous or noisy multimodal evidence
- complex reasoning with multiple uncertain assumptions
- prior attempts show disagreement or repeated wrong reasoning direction
- final attempt where reliability is critical

MODEL SELECTION POLICY:

Model selection should consider:
1) task type
2) task difficulty
3) model cost

Task-type guidance:
- CALCULATION tasks benefit from models that are strong at mathematical or symbolic reasoning.
- SCI_REASONING tasks usually require the strongest reasoning models.
- VISION_REASONING tasks should prefer models with multimodal capability.

Difficulty adjustment:
- EASY -> default to lower-cost model.
- MEDIUM -> choose a balanced model.
- HARD -> default to the strongest available model.

Failure-aware adjustment:
- If previous failure is mainly reasoning quality, upshift to stronger model.
- If reasoning direction repeatedly fails, switch to a stronger or different model.

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
  "task_type": "CALCULATION|SCI_REASONING|VISION_REASONING",
  "difficulty": "EASY|MEDIUM|HARD",
  "reasoning": "short rationale tied to attempt history explaining remaining uncertainty",
  "model": "one of available sub-models (required if delegate_task)",
  "instruction": "specific reflection instruction describing what to verify next (required if delegate_task)",
  "submit_reason": "why current evidence is enough to finalize (required if submit)"
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
