from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.mas_orchestra.schema import OfflineReplaySample, OfflineReplayStepSpec
from rllm.data import Dataset

DEFAULT_OFFLINE_DATASET_NAME = "sgi_reasoning_mas_orchestra_offline_replay"


def _format_location(path: str | Path | None, row_idx: int | None = None, step_idx: int | None = None) -> str:
    location = str(path) if path is not None else "offline replay row"
    if row_idx is not None:
        location += f" row {row_idx}"
    if step_idx is not None:
        location += f" step {step_idx}"
    return location


def _require_mapping(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object, got {type(value).__name__}")
    return value


def _require_messages(value: Any, *, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{location}.messages must be a list of message objects")

    messages: list[dict[str, Any]] = []
    for message_idx, message in enumerate(value):
        message_location = f"{location}.messages[{message_idx}]"
        message_dict = _require_mapping(message, location=message_location)
        if "role" not in message_dict:
            raise ValueError(f"{message_location} is missing required key 'role'")
        if "content" not in message_dict:
            raise ValueError(f"{message_location} is missing required key 'content'")
        messages.append(dict(message_dict))
    return messages


def _coerce_bool_like(value: Any, *, location: str) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, bool):
            return item
    raise ValueError(f"{location} must be a boolean")


def _require_bool_or_none(value: Any, *, location: str) -> bool | None:
    if value is None:
        return None
    return _coerce_bool_like(value, location=location)


def validate_offline_replay_step(step: Any, *, source_path: str | Path | None = None, row_idx: int | None = None, step_idx: int | None = None) -> OfflineReplayStepSpec:
    location = _format_location(source_path, row_idx=row_idx, step_idx=step_idx)
    step_dict = _require_mapping(step, location=location)

    required_keys = ("messages", "response", "step_reward", "trainable")
    for key in required_keys:
        if key not in step_dict:
            raise ValueError(f"{location} is missing required key '{key}'")

    metadata = step_dict.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{location}.metadata must be an object when provided")

    step_type = step_dict.get("step_type")
    if step_type is not None:
        step_type = str(step_type)

    model = step_dict.get("model")
    if model is not None:
        model = str(model)

    trainable = _coerce_bool_like(step_dict["trainable"], location=f"{location}.trainable")

    if step_dict["response"] is None:
        raise ValueError(f"{location}.response must be a string")

    return OfflineReplayStepSpec(
        messages=_require_messages(step_dict["messages"], location=location),
        response=str(step_dict["response"]),
        step_reward=float(step_dict["step_reward"]),
        trainable=trainable,
        step_type=step_type,
        model=model,
        metadata=dict(metadata),
    )


def validate_offline_replay_row(row: Any, *, source_path: str | Path | None = None, row_idx: int | None = None) -> OfflineReplaySample:
    location = _format_location(source_path, row_idx=row_idx)
    row_dict = _require_mapping(row, location=location)

    required_keys = ("task_id", "trajectory_reward", "steps")
    for key in required_keys:
        if key not in row_dict:
            raise ValueError(f"{location} is missing required key '{key}'")

    steps_value = row_dict["steps"]
    if not isinstance(steps_value, list) or not steps_value:
        raise ValueError(f"{location}.steps must be a non-empty list")

    steps = [
        validate_offline_replay_step(step, source_path=source_path, row_idx=row_idx, step_idx=step_idx)
        for step_idx, step in enumerate(steps_value)
    ]
    if not any(step.trainable for step in steps):
        raise ValueError(f"{location} must contain at least one trainable step")

    data_source = row_dict.get("data_source")
    if data_source is not None:
        data_source = str(data_source)

    if row_dict["task_id"] is None:
        raise ValueError(f"{location}.task_id must be a string")

    return OfflineReplaySample(
        task_id=str(row_dict["task_id"]),
        data_source=data_source,
        is_correct=_require_bool_or_none(row_dict.get("is_correct"), location=f"{location}.is_correct"),
        trajectory_reward=float(row_dict["trajectory_reward"]),
        steps=steps,
    )


def sample_to_row(sample: OfflineReplaySample) -> dict[str, Any]:
    return asdict(sample)


def validate_offline_replay_rows(rows: list[dict[str, Any]], *, source_path: str | Path | None = None) -> list[dict[str, Any]]:
    validated_rows = []
    for row_idx, row in enumerate(rows):
        validated_rows.append(sample_to_row(validate_offline_replay_row(row, source_path=source_path, row_idx=row_idx)))
    return validated_rows


def load_offline_replay_file(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        shard_paths = sorted(
            candidate
            for pattern in ("*.parquet", "*.arrow", "*.jsonl", "*.json")
            for candidate in resolved.glob(pattern)
            if candidate.is_file()
        )
        if not shard_paths:
            raise FileNotFoundError(f"No offline replay files found in directory: {resolved}")

        rows: list[dict[str, Any]] = []
        for shard_path in shard_paths:
            dataset = Dataset.load_data(str(shard_path))
            rows.extend(dataset.get_data())
        return validate_offline_replay_rows(rows, source_path=resolved)

    dataset = Dataset.load_data(str(resolved))
    return validate_offline_replay_rows(dataset.get_data(), source_path=resolved)
