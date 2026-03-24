from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")
CONFIDENCE_PATTERN = re.compile(r"confidence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_json_fragment(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    candidates: list[str] = []
    block = JSON_BLOCK_PATTERN.search(text)
    if block:
        candidates.append(block.group(1).strip())

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload

    return None


def extract_boxed_letters(text: str) -> list[str]:
    if not text:
        return []
    return [match.upper() for match in BOXED_PATTERN.findall(text)]


def extract_unique_boxed_letter(text: str) -> tuple[str | None, str | None]:
    letters = extract_boxed_letters(text)
    if not letters:
        return None, "No boxed answer found"

    unique = sorted(set(letters))
    if len(unique) > 1:
        return None, f"Conflicting boxed answers: {unique}"

    return unique[0], None


def clamp_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def extract_confidence(raw_text: str, json_payload: dict[str, Any] | None) -> float | None:
    if json_payload and "confidence" in json_payload:
        try:
            return clamp_confidence(float(json_payload["confidence"]))
        except Exception:
            pass

    if not raw_text:
        return None

    match = CONFIDENCE_PATTERN.search(raw_text)
    if match:
        try:
            value = float(match.group(1))
            if value > 1:
                value = value / 100.0
            return clamp_confidence(value)
        except Exception:
            return None

    percent_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", raw_text)
    if percent_match:
        try:
            return clamp_confidence(float(percent_match.group(1)) / 100.0)
        except Exception:
            return None

    return None
