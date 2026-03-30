from __future__ import annotations

import json

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from verl.utils.dataset.rl_dataset import RLHFDataset


class LocalParquetRLHFDataset(RLHFDataset):
    """RLHFDataset variant that avoids fragile nested parquet loading paths.

    Some pyarrow/datasets combinations fail when scanning nested parquet columns
    through ``datasets.load_dataset("parquet", ...)`` or direct pyarrow parquet
    readers. This loader falls back to pandas parquet loading when pyarrow hits
    nested/chunked conversion limits, and it can also read json/jsonl verl
    companions directly while keeping the rest of VERL's preprocessing logic
    unchanged.
    """

    @staticmethod
    def _normalize_loaded_value(value):
        if isinstance(value, dict):
            return {key: LocalParquetRLHFDataset._normalize_loaded_value(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [LocalParquetRLHFDataset._normalize_loaded_value(item) for item in value]
        if isinstance(value, np.ndarray):
            return [LocalParquetRLHFDataset._normalize_loaded_value(item) for item in value.tolist()]
        if hasattr(value, "item"):
            try:
                item = value.item()
            except Exception:
                item = value
            if item is not value:
                return LocalParquetRLHFDataset._normalize_loaded_value(item)
        return value

    @classmethod
    def _rows_from_pandas(cls, parquet_file: str) -> list[dict]:
        dataframe = pd.read_parquet(parquet_file)
        rows = dataframe.to_dict("records")
        return [cls._normalize_loaded_value(row) for row in rows]

    @classmethod
    def _rows_from_pyarrow(cls, parquet_file: str) -> list[dict]:
        table = pq.read_table(parquet_file)
        return [cls._normalize_loaded_value(row) for row in table.to_pylist()]

    @classmethod
    def _rows_from_json(cls, data_file: str) -> list[dict]:
        if data_file.endswith(".jsonl"):
            with open(data_file, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            with open(data_file, encoding="utf-8") as f:
                rows = json.load(f)
        return [cls._normalize_loaded_value(row) for row in rows]

    def _read_files_and_tokenize(self):
        dataframes = []
        data_files = self.data_files if isinstance(self.data_files, list | tuple) else [self.data_files]
        for parquet_file in data_files:
            if parquet_file.endswith(".json") or parquet_file.endswith(".jsonl"):
                rows = self._rows_from_json(parquet_file)
            else:
                try:
                    rows = self._rows_from_pyarrow(parquet_file)
                except Exception:
                    rows = self._rows_from_pandas(parquet_file)
            dataframes.append(datasets.Dataset.from_list(rows))

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
