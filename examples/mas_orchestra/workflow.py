from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

from PIL import Image

from examples.mas_orchestra.memory import MainMemory
from examples.mas_orchestra.parsing import parse_json_fragment
from examples.mas_orchestra.prompts import build_main_prompt, build_sub_prompt
from examples.mas_orchestra.scoring import compute_mca
from examples.mas_orchestra.subclients import MockSubModelClient, SubModelClient, build_delegate_request, build_delegate_result, task_to_reasoning_sample
from examples.mas_orchestra.types import AttemptRecord, MainAction, ReasoningSample
from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.workflows.workflow import TerminationReason, Workflow

DEFAULT_DATASET_NAME = "sgi_reasoning_mas_orchestra"
DEFAULT_WORKFLOW_ARGS = {
    "main_model": "local-policy",
    "sub_models": ["remote-submodel", "local-policy"],
    "max_attempts": 3,
    "submit_confidence_threshold": 0.75,
    "temperature_main": 1.0,
    "temperature_sub": 1.0,
    "use_images": True,
    "mock_external_submodels": True,
    "mock_delegate_responses": None,
}


def normalize_images(images: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for image in images:
        if isinstance(image, Image.Image):
            normalized.append(image)
        elif isinstance(image, dict) and "bytes" in image:
            normalized.append(Image.open(BytesIO(image["bytes"])))
        elif isinstance(image, bytes):
            normalized.append(Image.open(BytesIO(image)))
        else:
            normalized.append(image)
    return normalized


def select_sub_model(sub_models: list[str], attempt_offset: int) -> str:
    if not sub_models:
        raise ValueError("sub_models must contain at least one model name")
    return sub_models[attempt_offset % len(sub_models)]


def parse_main_action(raw_text: str) -> MainAction:
    payload = parse_json_fragment(raw_text)
    if not payload:
        return MainAction(action="invalid", reasoning="Failed to parse JSON")

    return MainAction(
        action=str(payload.get("action", "invalid")).strip(),
        reasoning=str(payload.get("reasoning", "")).strip(),
        model=str(payload["model"]) if payload.get("model") is not None else None,
        instruction=str(payload["instruction"]) if payload.get("instruction") is not None else None,
        submit_reason=str(payload["submit_reason"]) if payload.get("submit_reason") is not None else None,
    )


def should_submit(memory: MainMemory, threshold: float) -> bool:
    best = memory.best_attempt()
    if best is None:
        return False
    if not best.delegate_result.parse_ok:
        return False
    if best.delegate_result.confidence is None:
        return False
    return best.delegate_result.confidence >= threshold


def build_submit_result(memory: MainMemory, reason: str) -> dict[str, Any]:
    if not memory.attempts:
        return {
            "final_answer_text": "",
            "final_boxed_letter": None,
            "done": True,
            "reason": reason or "No attempts available; empty submission.",
            "attempt_count": 0,
        }

    parseable = [
        attempt
        for attempt in memory.attempts
        if attempt.delegate_result.parse_ok and attempt.delegate_result.boxed_letter is not None
    ]
    best = max(
        parseable or memory.attempts[-1:],
        key=lambda item: item.delegate_result.confidence if item.delegate_result.confidence is not None else -1.0,
    )
    return {
        "final_answer_text": best.delegate_result.raw_answer_text,
        "final_boxed_letter": best.delegate_result.boxed_letter,
        "done": True,
        "reason": reason or "Submitted best available attempt.",
        "attempt_count": len(memory.attempts),
    }


class MasOrchestraWorkflow(Workflow):
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        main_model: str,
        sub_models: list[str],
        executor=None,
        max_attempts: int = 3,
        submit_confidence_threshold: float = 0.75,
        temperature_main: float = 1.0,
        temperature_sub: float = 1.0,
        use_images: bool = True,
        mock_external_submodels: bool = True,
        mock_delegate_responses: dict[str, list[str] | str] | list[str] | str | None = None,
        sub_model_client: SubModelClient | None = None,
        **kwargs,
    ):
        super().__init__(
            rollout_engine=rollout_engine,
            executor=executor or ThreadPoolExecutor(max_workers=1),
            **kwargs,
        )
        self.main_model = main_model
        self.sub_models = sub_models
        self.max_attempts = max_attempts
        self.submit_confidence_threshold = submit_confidence_threshold
        self.temperature_main = temperature_main
        self.temperature_sub = temperature_sub
        self.use_images = use_images
        if sub_model_client is not None:
            self.sub_model_client = sub_model_client
        elif mock_external_submodels:
            self.sub_model_client = MockSubModelClient(mock_delegate_responses)
        else:
            raise ValueError("This example only supports mocked external sub-models. Provide a sub_model_client or enable mock_external_submodels.")

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)
        sample = task_to_reasoning_sample(task)
        sample.images = normalize_images(sample.images)
        memory = MainMemory()
        trajectory = Trajectory(name="orchestra_policy", task=task)
        submit_result: dict[str, Any] | None = None
        model_usage: dict[str, int] = {}
        models_used: list[str] = []
        main_policy_calls = 0
        delegate_policy_calls = 0
        external_delegate_calls = 0
        termination_reason = TerminationReason.ENV_DONE

        def record_model_call(model_name: str) -> None:
            model_usage[model_name] = model_usage.get(model_name, 0) + 1
            if model_name not in models_used:
                models_used.append(model_name)

        for attempt_idx in range(1, self.max_attempts + 1):
            force_submit = attempt_idx == self.max_attempts and bool(memory.attempts)
            main_messages = self._build_main_messages(sample, memory, attempt_idx)
            main_output = await self.rollout_engine.get_model_response(
                main_messages,
                application_id=f"{uid}:main:{attempt_idx}",
                temperature=self.temperature_main,
                **kwargs,
            )
            main_policy_calls += 1
            record_model_call(self.main_model)

            main_text = main_output.content or main_output.text or ""
            parsed_action = parse_main_action(main_text)
            if force_submit and should_submit(memory, self.submit_confidence_threshold):
                parsed_action = MainAction(
                    action="submit",
                    reasoning="Force submit at final attempt.",
                    submit_reason="Reached final allowed attempt; submitting best parseable candidate.",
                )
            elif parsed_action.action == "submit" and not should_submit(memory, self.submit_confidence_threshold):
                parsed_action = MainAction(
                    action="delegate_task",
                    reasoning="Submit rejected by guardrail; confidence below threshold.",
                    model=select_sub_model(self.sub_models, len(memory.attempts)),
                    instruction="Re-check ambiguous evidence and return one boxed answer with confidence.",
                )
            elif parsed_action.action == "submit":
                pass
            elif parsed_action.action == "delegate_task":
                if parsed_action.model not in self.sub_models:
                    parsed_action.model = select_sub_model(self.sub_models, len(memory.attempts))
                if not parsed_action.instruction:
                    parsed_action.instruction = "Refine reasoning and avoid earlier mistakes."
            else:
                parsed_action = MainAction(
                    action="delegate_task",
                    reasoning="Malformed action from orchestrator output; fallback to delegate.",
                    model=select_sub_model(self.sub_models, len(memory.attempts)),
                    instruction="Retry with strict format and clearer confidence.",
                )

            main_step = Step.from_model_output(
                main_output,
                messages=main_messages,
                action=Action(
                    action={
                        "action": parsed_action.action,
                        "reasoning": parsed_action.reasoning,
                        "model": parsed_action.model,
                        "instruction": parsed_action.instruction,
                        "submit_reason": parsed_action.submit_reason,
                    }
                ),
            )
            trajectory.steps.append(main_step)

            if main_output.finish_reason == "length":
                termination_reason = TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
                break

            if parsed_action.action == "submit":
                submit_result = build_submit_result(memory, parsed_action.submit_reason or parsed_action.reasoning)
                break

            request = build_delegate_request(
                sample,
                memory.attempts,
                parsed_action.model or select_sub_model(self.sub_models, len(memory.attempts)),
                parsed_action.instruction or "Refine reasoning and avoid earlier mistakes.",
            )

            if request.model == self.main_model:
                delegate_messages = self._build_delegate_messages(sample, memory, request.instruction)
                delegate_output = await self.rollout_engine.get_model_response(
                    delegate_messages,
                    application_id=f"{uid}:delegate:{attempt_idx}",
                    temperature=self.temperature_sub,
                    **kwargs,
                )
                delegate_policy_calls += 1
                record_model_call(self.main_model)
                delegate_text = delegate_output.content or delegate_output.text or ""
                delegate_result = build_delegate_result(
                    delegate_text,
                    input_tokens=len(delegate_output.prompt_ids or []),
                    output_tokens=len(delegate_output.completion_ids or []),
                )
                delegate_step = Step.from_model_output(
                    delegate_output,
                    messages=delegate_messages,
                    action=Action(
                        action={
                            "action": "self_think",
                            "model": request.model,
                            "instruction": request.instruction,
                        }
                    ),
                )
                trajectory.steps.append(delegate_step)
                if delegate_output.finish_reason == "length":
                    termination_reason = TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
                    break
            else:
                external_delegate_calls += 1
                record_model_call(request.model)
                delegate_result = await self.sub_model_client.run(request)

            memory.add_attempt(
                AttemptRecord(
                    attempt_index=attempt_idx,
                    model=request.model,
                    instruction=request.instruction,
                    delegate_result=delegate_result,
                    main_reasoning=parsed_action.reasoning,
                )
            )

        if submit_result is None:
            submit_result = build_submit_result(memory, "Fallback submit after max attempts.")

        mca = compute_mca(sample, submit_result["final_answer_text"])
        if trajectory.steps:
            trajectory.steps[-1].reward = mca
            trajectory.steps[-1].done = True
        trajectory.metadata = {
            "boxed_letter": submit_result["final_boxed_letter"],
            "attempt_count": submit_result["attempt_count"],
        }

        episode = Episode(
            task=task,
            trajectories=[trajectory],
            artifacts={
                "answer": submit_result["final_answer_text"],
                "boxed_letter": submit_result["final_boxed_letter"],
            },
            metrics={},
            metadata={},
        )
        episode = self.postprocess_episode(episode, termination_reason)
        episode.metrics.update(
            {
                "mca": mca,
                "attempt_count": submit_result["attempt_count"],
                "main_policy_calls": main_policy_calls,
                "delegate_policy_calls": delegate_policy_calls,
                "external_delegate_calls": external_delegate_calls,
            }
        )
        episode.metadata.update(
            {
                "models_used": models_used,
                "model_usage": model_usage,
                "submit_reason": submit_result["reason"],
            }
        )
        return episode

    def _build_main_messages(self, sample: ReasoningSample, memory: MainMemory, attempt_index: int) -> list[dict[str, Any]]:
        prompt = build_main_prompt(
            sample=sample,
            attempt_history_text=memory.as_brief_text(),
            attempt_index=attempt_index,
            max_attempts=self.max_attempts,
            sub_models=self.sub_models,
            threshold=self.submit_confidence_threshold,
        )
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if self.use_images and sample.images:
            user_message["images"] = sample.images
        return [
            {"role": "system", "content": "You are a strict orchestration controller. Output JSON only."},
            user_message,
        ]

    def _build_delegate_messages(self, sample: ReasoningSample, memory: MainMemory, instruction: str) -> list[dict[str, Any]]:
        prompt = build_sub_prompt(sample=sample, prior_attempts=memory.attempts, instruction=instruction)
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if sample.images:
            user_message["images"] = sample.images
        return [
            {"role": "system", "content": "You are a rigorous multimodal scientific reasoning assistant."},
            user_message,
        ]
