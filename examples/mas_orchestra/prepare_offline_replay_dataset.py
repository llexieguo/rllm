from __future__ import annotations

import argparse
from pathlib import Path

from examples.mas_orchestra.offline_replay import DEFAULT_OFFLINE_DATASET_NAME, load_offline_replay_file
from rllm.data.dataset import DatasetRegistry


def _register_split(dataset_name: str, split: str, path: str):
    resolved = str(Path(path).expanduser().resolve())
    rows = load_offline_replay_file(resolved)
    return DatasetRegistry.register_dataset(
        dataset_name,
        rows,
        split,
        source=resolved,
        description=f"MAS Orchestra offline replay data from {resolved}.",
        category="offline-replay",
    )


def prepare_offline_replay_dataset(
    *,
    dataset_name: str = DEFAULT_OFFLINE_DATASET_NAME,
    train_file: str,
    val_file: str | None = None,
    test_file: str | None = None,
):
    train_dataset = _register_split(dataset_name, "train", train_file)
    val_dataset = _register_split(dataset_name, "val", val_file) if val_file is not None else None
    test_dataset = _register_split(dataset_name, "test", test_file) if test_file is not None else None
    return train_dataset, val_dataset, test_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Register canonical MAS Orchestra offline replay JSONL/Parquet files in DatasetRegistry.")
    parser.add_argument("--dataset-name", default=DEFAULT_OFFLINE_DATASET_NAME)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--test-file", default=None)
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = prepare_offline_replay_dataset(
        dataset_name=args.dataset_name,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {0 if val_dataset is None else len(val_dataset)}")
    print(f"Test dataset size: {0 if test_dataset is None else len(test_dataset)}")
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Validation dataset path: {None if val_dataset is None else val_dataset.get_data_path()}")
    print(f"Test dataset path: {None if test_dataset is None else test_dataset.get_data_path()}")


if __name__ == "__main__":
    main()
