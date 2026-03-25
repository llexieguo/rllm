from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import fcntl
from PIL import Image

from examples.mas_orchestra.parsing import extract_confidence, extract_unique_boxed_letter, parse_json_fragment
from examples.mas_orchestra.prompts import build_sub_prompt
from examples.mas_orchestra.schema import DelegateRequest, DelegateResult, ReasoningSample

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - dependency/environment dependent
    AsyncOpenAI = None


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _resolve_run_dir() -> Path:
    root = Path(os.environ.get("MAS_ORCHESTRA_STEP_LOG_ROOT", "logs/mas_orchestra")).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    run_id = os.environ.get("MAS_ORCHESTRA_RUN_ID", "default_run")
    return root / run_id


def _pricing(model_name: str | None = None) -> tuple[float, float]:
    model_key = (model_name or "").upper().replace("/", "_").replace("-", "_")
    model_input = _env_float(f"MAS_ORCHESTRA_API_{model_key}_INPUT_COST_PER_1M", None)
    model_output = _env_float(f"MAS_ORCHESTRA_API_{model_key}_OUTPUT_COST_PER_1M", None)
    if model_input is not None or model_output is not None:
        return float(model_input or 0.0), float(model_output or 0.0)
    return (
        float(_env_float("MAS_ORCHESTRA_API_INPUT_COST_PER_1M", 0.0) or 0.0),
        float(_env_float("MAS_ORCHESTRA_API_OUTPUT_COST_PER_1M", 0.0) or 0.0),
    )


def _compute_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    input_cost_per_1m, output_cost_per_1m = _pricing(model_name)
    return (input_tokens / 1_000_000.0) * input_cost_per_1m + (output_tokens / 1_000_000.0) * output_cost_per_1m


def _image_to_data_url(image: Any) -> str:
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        pil_image = Image.open(BytesIO(image))
    else:
        raise TypeError(f"Unsupported image type for external API client: {type(image)!r}")

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text") and item.text is not None:
                parts.append(str(item.text))
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return "" if content is None else str(content)


def build_delegate_result(
    raw_text: str,
    *,
    cost: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    provider_model: str | None = None,
    response_id: str | None = None,
    running_total_cost: float | None = None,
    cost_budget: float | None = None,
    budget_exceeded: bool = False,
) -> DelegateResult:
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
        provider_model=provider_model,
        response_id=response_id,
        running_total_cost=running_total_cost,
        cost_budget=cost_budget,
        budget_exceeded=budget_exceeded,
    )


def delegate_result_to_log(result: DelegateResult) -> dict[str, Any]:
    return {
        "raw_answer_text": result.raw_answer_text,
        "boxed_letter": result.boxed_letter,
        "confidence": result.confidence,
        "reasoning_summary": result.reasoning_summary,
        "parse_ok": result.parse_ok,
        "error": result.error,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost": result.cost,
        "provider_model": result.provider_model,
        "response_id": result.response_id,
        "running_total_cost": result.running_total_cost,
        "cost_budget": result.cost_budget,
        "budget_exceeded": result.budget_exceeded,
    }


class ApiCostTracker:
    def __init__(self, budget: float | None = None, state_path: Path | None = None):
        self.budget = budget
        self.state_path = state_path or (_resolve_run_dir() / "api_budget_state.json")

    def record_call(self, *, model: str, cost: float, input_tokens: int, output_tokens: int) -> dict[str, Any]:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self.state_path.write_text("", encoding="utf-8")

        with self.state_path.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read().strip()
            state = json.loads(raw) if raw else {
                "budget": self.budget,
                "total_cost": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "call_count": 0,
            }
            state["budget"] = self.budget
            state["total_cost"] += cost
            state["total_input_tokens"] += input_tokens
            state["total_output_tokens"] += output_tokens
            state["call_count"] += 1
            state["last_model"] = model
            state["budget_exceeded"] = self.budget is not None and state["total_cost"] > self.budget
            handle.seek(0)
            handle.truncate()
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return state


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
            f'"final_answer":"\\boxed{{{default_letter}}}","confidence":0.61}}'
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
        return build_delegate_result(raw_text, provider_model=request.model)


class OpenAISubModelClient(SubModelClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        default_model: str | None = None,
        timeout: float = 120.0,
        system_prompt: str = "You are a rigorous multimodal scientific reasoning assistant.",
        cost_tracker: ApiCostTracker | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout) if AsyncOpenAI is not None else None
        self.default_model = default_model
        self.system_prompt = system_prompt
        self.cost_tracker = cost_tracker or ApiCostTracker(budget=_env_float("MAS_ORCHESTRA_API_COST_BUDGET", None))
        self.call_log: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls) -> "OpenAISubModelClient":
        base_url = os.environ.get("MAS_ORCHESTRA_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("base_url")
        api_key = os.environ.get("MAS_ORCHESTRA_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("api_key")
        if not base_url:
            raise ValueError("Missing MAS_ORCHESTRA_API_BASE_URL or OPENAI_BASE_URL for external submodel calls")
        if not api_key:
            raise ValueError("Missing MAS_ORCHESTRA_API_KEY or OPENAI_API_KEY for external submodel calls")
        return cls(
            base_url=base_url,
            api_key=api_key,
            default_model=os.environ.get("MAS_ORCHESTRA_API_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("model"),
            timeout=float(_env_float("MAS_ORCHESTRA_API_TIMEOUT", 120.0) or 120.0),
            cost_tracker=ApiCostTracker(budget=_env_float("MAS_ORCHESTRA_API_COST_BUDGET", None)),
        )

    def _build_messages(self, request: DelegateRequest) -> list[dict[str, Any]]:
        sample = ReasoningSample(
            task_id=request.task_id,
            question=request.question,
            options=request.options,
            answer_index=0,
            steps=[],
            discipline="unknown",
            images=request.images,
        )
        prompt = build_sub_prompt(sample=sample, prior_attempts=request.prior_attempts, instruction=request.instruction)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in request.images:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content if len(content) > 1 else prompt},
        ]

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib_request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:  # pragma: no cover - network dependent
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"External API HTTP error {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"External API connection error: {exc.reason}") from exc

    async def run(self, request: DelegateRequest) -> DelegateResult:
        request_model = request.model
        api_model = request_model
        if request_model in {"remote-submodel", "external-submodel", "api-submodel"} and self.default_model:
            api_model = self.default_model
        elif not api_model and self.default_model:
            api_model = self.default_model
        messages = self._build_messages(request)
        payload = {"model": api_model, "messages": messages}
        if self.client is not None:
            response = await self.client.chat.completions.create(model=api_model, messages=messages)
            response_json = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        else:
            response_json = await asyncio.to_thread(self._post_chat_completion, payload)

        choice = response_json["choices"][0]["message"]
        raw_text = _extract_text_content(choice.get("content"))
        usage = response_json.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        cost = _compute_cost(api_model, input_tokens, output_tokens)
        budget_state = self.cost_tracker.record_call(model=api_model, cost=cost, input_tokens=input_tokens, output_tokens=output_tokens)
        result = build_delegate_result(
            raw_text,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_model=response_json.get("model", api_model),
            response_id=response_json.get("id"),
            running_total_cost=budget_state["total_cost"],
            cost_budget=budget_state.get("budget"),
            budget_exceeded=bool(budget_state.get("budget_exceeded", False)),
        )
        self.call_log.append(
            {
                "task_id": request.task_id,
                "requested_model": request.model,
                "provider_model": api_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "running_total_cost": result.running_total_cost,
                "budget": result.cost_budget,
                "budget_exceeded": result.budget_exceeded,
            }
        )
        return result


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
