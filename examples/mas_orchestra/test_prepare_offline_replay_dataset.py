from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from examples.mas_orchestra.prepare_offline_replay_dataset import prepare_offline_replay_dataset
from rllm.data.dataset import DatasetRegistry, deserialize_verl_extra_info
from rllm.trainer.verl.local_parquet_rl_dataset import LocalParquetRLHFDataset


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
    assert Path(verl_path).suffix == ".jsonl"
    with Path(verl_path).open(encoding="utf-8") as handle:
        verl_row = json.loads(handle.readline())
    assert verl_row["extra_info"].keys() == {"__rllm_payload__"}
    decoded = deserialize_verl_extra_info(verl_row["extra_info"])
    assert decoded["task_id"] == "t1"
    assert decoded["steps"][0]["messages"][0]["content"] == "Question t1"

    dataset = LocalParquetRLHFDataset(
        data_files=verl_path,
        tokenizer=None,
        config=OmegaConf.create(
            {
                "cache_dir": str(tmp_path / "cache"),
                "filter_overlong_prompts": False,
                "return_multi_modal_inputs": False,
            }
        ),
    )
    assert len(dataset) == 2
    assert dataset.dataframe[0]["extra_info"]["__rllm_payload__"].startswith("pickle_b64:")


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


def test_local_parquet_rlhf_dataset_reads_nested_multi_row_group_parquet(tmp_path):
    rows = DatasetRegistry.apply_verl_postprocessing([build_row(f"rg-{idx}") for idx in range(5)])
    parquet_path = tmp_path / "nested_multi_row_group.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path, row_group_size=2)

    dataset = LocalParquetRLHFDataset(
        data_files=str(parquet_path),
        tokenizer=None,
        config=OmegaConf.create(
            {
                "cache_dir": str(tmp_path / "cache"),
                "filter_overlong_prompts": False,
                "return_multi_modal_inputs": False,
            }
        ),
    )

    assert len(dataset) == 5
    assert dataset.dataframe[0]["extra_info"]["__rllm_payload__"].startswith("pickle_b64:")


def test_local_parquet_rlhf_dataset_falls_back_to_pandas_for_nested_parquet(tmp_path, monkeypatch):
    rows = DatasetRegistry.apply_verl_postprocessing([build_row("fallback")])
    parquet_path = tmp_path / "nested_fallback.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path, row_group_size=1)

    monkeypatch.setattr(
        LocalParquetRLHFDataset,
        "_rows_from_pyarrow",
        classmethod(lambda cls, parquet_file: (_ for _ in ()).throw(RuntimeError("force pandas fallback"))),
    )

    dataset = LocalParquetRLHFDataset(
        data_files=str(parquet_path),
        tokenizer=None,
        config=OmegaConf.create(
            {
                "cache_dir": str(tmp_path / "cache"),
                "filter_overlong_prompts": False,
                "return_multi_modal_inputs": False,
            }
        ),
    )

    assert len(dataset) == 1
    assert dataset.dataframe[0]["prompt"][0]["role"] == "user"
