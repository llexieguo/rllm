from __future__ import annotations

from datasets import Dataset

from examples.mas_orchestra.prepare_dataset import prepare_dataset
from rllm.data.dataset import DatasetRegistry


def configure_registry(tmp_path, monkeypatch):
    dataset_root = tmp_path / ".rllm" / "datasets"
    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", str(tmp_path / ".rllm"))
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", str(dataset_root / "registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", str(dataset_root))


def test_prepare_dataset_registers_registry_and_verl_files(tmp_path, monkeypatch):
    configure_registry(tmp_path, monkeypatch)

    train_rows = [
        {
            "idx": 1,
            "question": "Q1",
            "options": ["A", "B", "C", "D"],
            "answer": 0,
            "steps": ["s1"],
            "discipline": "physics",
            "images": [],
        }
    ]
    val_rows = [
        {
            "idx": 2,
            "question": "Q2",
            "options": ["A", "B", "C", "D"],
            "answer": 1,
            "steps": ["s2"],
            "discipline": "chemistry",
            "images": [],
        }
    ]

    def fake_load_dataset(source_name: str, split: str):
        if split == "train":
            return Dataset.from_list(train_rows)
        return Dataset.from_list(val_rows)

    monkeypatch.setattr("examples.mas_orchestra.prepare_dataset.load_dataset", fake_load_dataset)

    train_dataset, val_dataset = prepare_dataset(
        dataset_name="sgi_reasoning_mas_test",
        source_name="mock-source",
    )

    assert train_dataset.get_data_path() is not None
    assert train_dataset.get_verl_data_path() is not None
    assert val_dataset.get_data_path() is not None
    assert val_dataset.get_verl_data_path() is not None
    assert train_dataset[0]["question"] == "Q1"
    assert val_dataset[0]["answer_index"] == 1
