from __future__ import annotations


class ModelPricing:
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
    def resolve_pricing(cls, model_name: str) -> dict[str, float] | None:
        if model_name in cls.PRICES:
            return cls.PRICES[model_name]
        for known, price in cls.PRICES.items():
            if known in model_name:
                return price
        return None
