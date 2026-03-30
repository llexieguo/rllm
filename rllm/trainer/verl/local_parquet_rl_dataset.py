from __future__ import annotations

import datasets
import numpy as np
import pyarrow.parquet as pq
from verl.utils.dataset.rl_dataset import RLHFDataset


class LocalParquetRLHFDataset(RLHFDataset):
    """RLHFDataset variant that avoids HF parquet scanning for nested columns.

    Some pyarrow/datasets combinations fail when scanning nested parquet columns
    through ``datasets.load_dataset("parquet", ...)``. Reading record batches via
    ``pyarrow.parquet.ParquetFile`` and then constructing a HuggingFace Dataset
    from Python rows avoids that code path while keeping the rest of VERL's
    preprocessing logic unchanged.
    """

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            parquet = pq.ParquetFile(parquet_file)
            rows: list[dict] = []
            for record_batch in parquet.iter_batches():
                rows.extend(record_batch.to_pylist())
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
