from __future__ import annotations

import argparse
from io import BytesIO
from typing import Any

from datasets import load_dataset
from PIL import Image

from examples.mas_orchestra.workflow import DEFAULT_DATASET_NAME
from rllm.data.dataset import DatasetRegistry


DEFAULT_SOURCE_NAME = "InternScience/SGI-Reasoning"
DEFAULT_SOURCE_SPLIT = "test"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 145


def _serialize_image(image: Any) -> Any:
    if isinstance(image, dict) and "bytes" in image:
        return image
    if isinstance(image, bytes):
        return {"bytes": image}
    if isinstance(image, Image.Image):
        buffer = BytesIO()
        image.save(buffer, format=image.format or "PNG")
        return {"bytes": buffer.getvalue()}
    return image


def build_reasoning_row(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(example.get("idx", example.get("id", ""))),
        "question": str(example.get("question", "")),
        "options": [str(option) for option in example.get("options", [])],
        "answer_index": int(example.get("answer", 0)),
        "steps": [str(step) for step in example.get("steps", [])],
        "discipline": str(example.get("discipline", "unknown")),
        "images": [_serialize_image(image) for image in (example.get("images", []) or [])],
        "data_source": "sgi_reasoning",
    }


def _select_limit(dataset, limit: int | None):
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _split_single_test_source(raw_dataset, *, test_size: int, val_size: int, seed: int):
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if val_size < 0:
        raise ValueError("val_size must be non-negative")
    if val_size >= test_size:
        raise ValueError("val_size must be smaller than test_size")
    if len(raw_dataset) <= test_size:
        raise ValueError(
            f"Source split must contain more than {test_size} rows so the remainder can be used for training; got {len(raw_dataset)} rows."
        )

    shuffled = raw_dataset.shuffle(seed=seed)
    eval_pool = shuffled.select(range(test_size))
    train_raw = shuffled.select(range(test_size, len(shuffled)))

    if val_size > 0:
        val_raw = eval_pool.select(range(val_size))
        test_raw = eval_pool.select(range(val_size, len(eval_pool)))
    else:
        val_raw = None
        test_raw = eval_pool

    return train_raw, test_raw, val_raw


def prepare_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    source_name: str = DEFAULT_SOURCE_NAME,
    source_split: str = DEFAULT_SOURCE_SPLIT,
    train_limit: int | None = None,
    test_size: int = DEFAULT_TEST_SIZE,
    val_size: int = 0,
    seed: int = DEFAULT_SEED,
):
    raw_dataset = load_dataset(source_name, split=source_split)
    train_raw, test_raw, val_raw = _split_single_test_source(raw_dataset, test_size=test_size, val_size=val_size, seed=seed)

    train_raw = _select_limit(train_raw, train_limit)

    train_rows = [build_reasoning_row(example) for example in train_raw]
    test_rows = [build_reasoning_row(example) for example in test_raw]
    val_rows = [build_reasoning_row(example) for example in val_raw] if val_raw is not None else None

    train_dataset = DatasetRegistry.register_dataset(
        dataset_name,
        train_rows,
        "train",
        source=source_name,
        description="MAS Orchestra workflow training data derived from the source test split.",
        category="multimodal-reasoning",
    )
    test_dataset = DatasetRegistry.register_dataset(
        dataset_name,
        test_rows,
        "test",
        source=source_name,
        description="MAS Orchestra workflow held-out test data derived from the source test split.",
        category="multimodal-reasoning",
    )

    val_dataset = None
    if val_rows is not None:
        val_dataset = DatasetRegistry.register_dataset(
            dataset_name,
            val_rows,
            "val",
            source=source_name,
            description="MAS Orchestra workflow optional validation data carved out of the held-out test pool.",
            category="multimodal-reasoning",
        )

    return train_dataset, test_dataset, val_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SGI-Reasoning for MAS Orchestra when the source dataset only provides a test split.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--source-split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    train_dataset, test_dataset, val_dataset = prepare_dataset(
        dataset_name=args.dataset_name,
        source_name=args.source_name,
        source_split=args.source_split,
        train_limit=args.train_limit,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Validation dataset size: {0 if val_dataset is None else len(val_dataset)}")
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Test dataset path: {test_dataset.get_data_path()}")
    print(f"Validation dataset path: {None if val_dataset is None else val_dataset.get_data_path()}")


if __name__ == "__main__":
    main()
