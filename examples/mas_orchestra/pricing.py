from __future__ import annotations

import os


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    return float(raw)


class ModelPricing:
    PLACEHOLDER_MODELS = {"remote-submodel", "external-submodel", "api-submodel"}
    PRICES = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4.1": {"input": 0.002, "output": 0.008},
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
        "o3": {"input": 0.002, "output": 0.008},
        "o3-mini": {"input": 0.0011, "output": 0.0044},
        "o4-mini": {"input": 0.0011, "output": 0.0044},
        "gpt-5": {"input": 0.00125, "output": 0.01},
        "gpt-5-mini": {"input": 0.00025, "output": 0.002},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-4-sonnet": {"input": 0.003, "output": 0.015},
        "claude-4-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-4-5-haiku": {"input": 0.00088, "output": 0.0044},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
        "gemini-2.5-flash": {"input": 0.0003, "output": 0.00252},
        "deepseek-chat": {"input": 0.00025, "output": 0.001},
        "deepseek-r1": {"input": 0.00055, "output": 0.00219},
    }

    @classmethod
    def resolve_delegate_target(cls, model_name: str) -> str:
        if model_name in cls.PLACEHOLDER_MODELS:
            for env_name in ("MAS_ORCHESTRA_API_MODEL", "OPENAI_MODEL", "model"):
                value = os.environ.get(env_name)
                if value:
                    return value
        return model_name

    @classmethod
    def format_model_label(cls, model_name: str) -> str:
        delegate_target = cls.resolve_delegate_target(model_name)
        if delegate_target != model_name:
            return f"{model_name} -> {delegate_target}"
        return model_name

    @classmethod
    def _env_pricing(cls, model_name: str) -> dict[str, float] | None:
        model_key = model_name.upper().replace("/", "_").replace("-", "_")
        model_input = _env_float(f"MAS_ORCHESTRA_API_{model_key}_INPUT_COST_PER_1M")
        model_output = _env_float(f"MAS_ORCHESTRA_API_{model_key}_OUTPUT_COST_PER_1M")
        if model_input is not None or model_output is not None:
            return {
                "input": float(model_input or 0.0) / 1000.0,
                "output": float(model_output or 0.0) / 1000.0,
            }

        generic_input = _env_float("MAS_ORCHESTRA_API_INPUT_COST_PER_1M")
        generic_output = _env_float("MAS_ORCHESTRA_API_OUTPUT_COST_PER_1M")
        if generic_input is not None or generic_output is not None:
            return {
                "input": float(generic_input or 0.0) / 1000.0,
                "output": float(generic_output or 0.0) / 1000.0,
            }
        return None

    @classmethod
    def resolve_pricing(cls, model_name: str) -> dict[str, float] | None:
        delegate_target = cls.resolve_delegate_target(model_name)
        env_price = cls._env_pricing(delegate_target)
        if env_price is not None:
            return env_price
        if delegate_target in cls.PRICES:
            return cls.PRICES[delegate_target]
        for known, price in cls.PRICES.items():
            if known in delegate_target:
                return price
        return None
