from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from examples.mas_orchestra.offline_replay import load_offline_replay_file


def split_offline_replay_file(*, input_file: str, output_dir: str, rows_per_file: int) -> list[Path]:
    if rows_per_file <= 0:
        raise ValueError("--rows-per-file must be positive")

    input_path = Path(input_file).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    rows = load_offline_replay_file(input_path)
    shard_paths: list[Path] = []
    for shard_idx, start in enumerate(range(0, len(rows), rows_per_file)):
        shard_rows = rows[start : start + rows_per_file]
        shard_path = output_path / f"part-{shard_idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(shard_rows), shard_path)
        shard_paths.append(shard_path)
    return shard_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a MAS Orchestra offline replay parquet into smaller parquet shards.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows-per-file", type=int, default=128)
    args = parser.parse_args()

    shard_paths = split_offline_replay_file(
        input_file=args.input_file,
        output_dir=args.output_dir,
        rows_per_file=args.rows_per_file,
    )

    print(f"Wrote {len(shard_paths)} parquet shard(s) to {Path(args.output_dir).expanduser().resolve()}")
    if shard_paths:
        print(f"First shard: {shard_paths[0]}")
        print(f"Last shard: {shard_paths[-1]}")


if __name__ == "__main__":
    main()
