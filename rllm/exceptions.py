from __future__ import annotations


class ExternalCostBudgetExceeded(RuntimeError):
    def __init__(self, *, total_cost: float, budget: float, model: str | None = None):
        self.total_cost = total_cost
        self.budget = budget
        self.model = model
        detail = f"External API cost budget exceeded: total_cost={total_cost:.6f}, budget={budget:.6f}"
        if model:
            detail += f", model={model}"
        super().__init__(detail)
