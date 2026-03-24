from __future__ import annotations

from datasets import Dataset

from examples.mas_orchestra.prepare_dataset import prepare_dataset
from rllm.data.dataset import DatasetRegistry


def configure_registry(tmp_path, monkeypatch):
    dataset_root = tmp_path / ".rllm" / "datasets"
    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", str(tmp_path / ".rllm"))
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", str(dataset_root / "registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", str(dataset_root))


def build_rows(num_rows: int) -> list[dict]:
    rows = []
    for idx in range(num_rows):
        rows.append(
            {
                "idx": idx,
                "question": f"Q{idx}",
                "options": ["A", "B", "C", "D"],
                "answer": idx % 4,
                "steps": [f"s{idx}"],
                "discipline": "physics",
                "images": [],
            }
        )
    return rows


def test_prepare_dataset_splits_single_test_source_without_validation(tmp_path, monkeypatch):
    configure_registry(tmp_path, monkeypatch)
    raw_rows = build_rows(200)

    def fake_load_dataset(source_name: str, split: str):
        assert source_name == "mock-source"
        assert split == "test"
        return Dataset.from_list(raw_rows)

    monkeypatch.setattr("examples.mas_orchestra.prepare_dataset.load_dataset", fake_load_dataset)

    train_dataset, test_dataset, val_dataset = prepare_dataset(
        dataset_name="sgi_reasoning_mas_test",
        source_name="mock-source",
        test_size=145,
        seed=42,
    )

    assert len(train_dataset) == 55
    assert len(test_dataset) == 145
    assert val_dataset is None
    assert train_dataset.get_data_path() is not None
    assert train_dataset.get_verl_data_path() is not None
    assert test_dataset.get_data_path() is not None
    assert test_dataset.get_verl_data_path() is not None

    train_ids = {row["task_id"] for row in train_dataset.get_data()}
    test_ids = {row["task_id"] for row in test_dataset.get_data()}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids | test_ids) == 200


def test_prepare_dataset_can_carve_validation_from_test_pool(tmp_path, monkeypatch):
    configure_registry(tmp_path, monkeypatch)
    raw_rows = build_rows(200)

    def fake_load_dataset(source_name: str, split: str):
        return Dataset.from_list(raw_rows)

    monkeypatch.setattr("examples.mas_orchestra.prepare_dataset.load_dataset", fake_load_dataset)

    _, test_dataset, val_dataset = prepare_dataset(
        dataset_name="sgi_reasoning_mas_val_test",
        source_name="mock-source",
        test_size=145,
        val_size=5,
        seed=42,
    )

    assert len(test_dataset) == 140
    assert val_dataset is not None
    assert len(val_dataset) == 5

    test_ids = {row["task_id"] for row in test_dataset.get_data()}
    val_ids = {row["task_id"] for row in val_dataset.get_data()}
    assert test_ids.isdisjoint(val_ids)
