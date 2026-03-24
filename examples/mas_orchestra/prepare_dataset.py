from __future__ import annotations

import argparse
from typing import Any

from datasets import load_dataset

from examples.mas_orchestra.workflow import DEFAULT_DATASET_NAME
from rllm.data.dataset import DatasetRegistry


def build_reasoning_row(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(example.get("idx", example.get("id", ""))),
        "question": str(example.get("question", "")),
        "options": [str(option) for option in example.get("options", [])],
        "answer_index": int(example.get("answer", 0)),
        "steps": [str(step) for step in example.get("steps", [])],
        "discipline": str(example.get("discipline", "unknown")),
        "images": list(example.get("images", []) or []),
        "data_source": "sgi_reasoning",
    }


def prepare_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    source_name: str = "InternScience/SGI-Reasoning",
    train_split: str = "train",
    val_split: str = "test",
    train_limit: int | None = None,
    val_limit: int | None = None,
):
    train_raw = load_dataset(source_name, split=train_split)
    val_raw = load_dataset(source_name, split=val_split)

    if train_limit is not None:
        train_raw = train_raw.select(range(min(train_limit, len(train_raw))))
    if val_limit is not None:
        val_raw = val_raw.select(range(min(val_limit, len(val_raw))))

    train_rows = [build_reasoning_row(example) for example in train_raw]
    val_rows = [build_reasoning_row(example) for example in val_raw]

    train_dataset = DatasetRegistry.register_dataset(
        dataset_name,
        train_rows,
        "train",
        source=source_name,
        description="MAS Orchestra workflow training data",
        category="multimodal-reasoning",
    )
    val_dataset = DatasetRegistry.register_dataset(
        dataset_name,
        val_rows,
        "test",
        source=source_name,
        description="MAS Orchestra workflow validation data",
        category="multimodal-reasoning",
    )
    return train_dataset, val_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SGI-Reasoning dataset for MAS Orchestra workflow training.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--source-name", default="InternScience/SGI-Reasoning")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    args = parser.parse_args()

    train_dataset, val_dataset = prepare_dataset(
        dataset_name=args.dataset_name,
        source_name=args.source_name,
        train_split=args.train_split,
        val_split=args.val_split,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Validation dataset path: {val_dataset.get_data_path()}")


if __name__ == "__main__":
    main()
