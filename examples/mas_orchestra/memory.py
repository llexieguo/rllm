from __future__ import annotations

from dataclasses import dataclass, field

from examples.mas_orchestra.schema import AttemptRecord


@dataclass
class MainMemory:
    attempts: list[AttemptRecord] = field(default_factory=list)

    def add_attempt(self, attempt: AttemptRecord) -> None:
        self.attempts.append(attempt)

    def best_attempt(self) -> AttemptRecord | None:
        if not self.attempts:
            return None

        parseable = [
            attempt
            for attempt in self.attempts
            if attempt.delegate_result.parse_ok and attempt.delegate_result.boxed_letter is not None
        ]
        if not parseable:
            return self.attempts[-1]

        return max(
            parseable,
            key=lambda item: item.delegate_result.confidence if item.delegate_result.confidence is not None else -1.0,
        )

    def as_brief_text(self) -> str:
        if not self.attempts:
            return "No attempts yet."

        lines: list[str] = []
        for attempt in self.attempts:
            delegate = attempt.delegate_result
            lines.append(
                (
                    f"Attempt {attempt.attempt_index} | model={attempt.model} "
                    f"| boxed={delegate.boxed_letter} "
                    f"| confidence={delegate.confidence} "
                    f"| parse_ok={delegate.parse_ok} "
                    f"| error={delegate.error or '-'}"
                )
            )
            if delegate.reasoning_summary:
                lines.append(f"Summary: {delegate.reasoning_summary}")
        return "\n".join(lines)
