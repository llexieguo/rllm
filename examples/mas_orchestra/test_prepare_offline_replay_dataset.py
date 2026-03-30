from __future__ import annotations

import json

import pandas as pd
import pytest

from examples.mas_orchestra.prepare_offline_replay_dataset import prepare_offline_replay_dataset
from rllm.data.dataset import DatasetRegistry, deserialize_verl_extra_info


def configure_registry(tmp_path, monkeypatch):
    dataset_root = tmp_path / ".rllm" / "datasets"
    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", str(tmp_path / ".rllm"))
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", str(dataset_root / "registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", str(dataset_root))


def build_row(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "data_source": "mas_orchestra_offline",
        "is_correct": True,
        "trajectory_reward": 3.0,
        "steps": [
            {
                "messages": [{"role": "user", "content": f"Question {task_id}"}],
                "response": "answer",
                "step_reward": 1.0,
                "trainable": True,
                "step_type": "main",
                "model": "local-policy",
                "metadata": {"attempt": 1},
            }
        ],
    }


def write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_prepare_offline_replay_dataset_registers_local_splits(tmp_path, monkeypatch):
    configure_registry(tmp_path, monkeypatch)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(train_path, [build_row("t1"), build_row("t2")])
    write_jsonl(val_path, [build_row("v1")])

    train_dataset, val_dataset, test_dataset = prepare_offline_replay_dataset(
        dataset_name="mas_offline_test",
        train_file=str(train_path),
        val_file=str(val_path),
    )

    assert len(train_dataset) == 2
    assert val_dataset is not None
    assert len(val_dataset) == 1
    assert test_dataset is None
    assert train_dataset.get_data_path() is not None
    assert train_dataset.get_verl_data_path() is not None

    loaded_train = DatasetRegistry.load_dataset("mas_offline_test", "train")
    assert loaded_train is not None
    assert loaded_train.get_data()[0]["steps"][0]["response"] == "answer"

    verl_path = train_dataset.get_verl_data_path()
    assert verl_path is not None
    verl_row = pd.read_parquet(verl_path).iloc[0].to_dict()
    assert verl_row["extra_info"].keys() == {"__rllm_payload__"}
    decoded = deserialize_verl_extra_info(verl_row["extra_info"])
    assert decoded["task_id"] == "t1"
    assert decoded["steps"][0]["messages"][0]["content"] == "Question t1"


def test_prepare_offline_replay_dataset_rejects_missing_row_field(tmp_path, monkeypatch):
    configure_registry(tmp_path, monkeypatch)
    bad_path = tmp_path / "bad.jsonl"
    bad_row = build_row("bad")
    del bad_row["trajectory_reward"]
    write_jsonl(bad_path, [bad_row])

    with pytest.raises(ValueError, match="trajectory_reward"):
        prepare_offline_replay_dataset(
            dataset_name="mas_offline_bad",
            train_file=str(bad_path),
        )


def test_prepare_offline_replay_dataset_rejects_missing_step_field(tmp_path, monkeypatch):
    configure_registry(tmp_path, monkeypatch)
    bad_path = tmp_path / "bad_step.jsonl"
    bad_row = build_row("bad-step")
    del bad_row["steps"][0]["response"]
    write_jsonl(bad_path, [bad_row])

    with pytest.raises(ValueError, match="response"):
        prepare_offline_replay_dataset(
            dataset_name="mas_offline_bad_step",
            train_file=str(bad_path),
        )


@pytest.mark.parametrize("missing_key", ["messages", "step_reward"])
def test_prepare_offline_replay_dataset_rejects_other_missing_step_fields(tmp_path, monkeypatch, missing_key):
    configure_registry(tmp_path, monkeypatch)
    bad_path = tmp_path / f"bad_{missing_key}.jsonl"
    bad_row = build_row(f"bad-{missing_key}")
    del bad_row["steps"][0][missing_key]
    write_jsonl(bad_path, [bad_row])

    with pytest.raises(ValueError, match=missing_key):
        prepare_offline_replay_dataset(
            dataset_name=f"mas_offline_bad_{missing_key}",
            train_file=str(bad_path),
        )
