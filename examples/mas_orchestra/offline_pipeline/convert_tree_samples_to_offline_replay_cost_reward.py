from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from examples.mas_orchestra.offline_pipeline.convert_tree_samples_to_offline_replay import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_SOURCE_DATASET,
    _fallback_task,
    _load_source_tasks,
    _reconstruct_main_steps,
    _safe_len,
    _select_trajectories,
    write_output,
)
from examples.mas_orchestra.offline_replay import validate_offline_replay_rows


def _trajectory_accuracy_reward(trajectory: dict[str, Any], incorrect_reward: float) -> float:
    return 1.0 if trajectory.get("correct") else incorrect_reward


def _trajectory_total_api_cost(node_ids: list[str], node_map: dict[str, dict[str, Any]]) -> float:
    total_cost = 0.0
    for node_id in node_ids:
        if node_id == "root":
            continue

        node = node_map.get(node_id, {})
        for attempt in node.get("orchestra_attempts", []):
            payload = attempt.get("delegate_result")
            if isinstance(payload, dict):
                total_cost += float(payload.get("cost", 0.0) or 0.0)
    return total_cost


def _annotate_steps_with_reward_metadata(
    steps: list[dict[str, Any]],
    *,
    accuracy_reward: float,
    total_api_cost: float,
    api_cost_coef: float,
    composite_reward: float,
) -> list[dict[str, Any]]:
    annotated_steps: list[dict[str, Any]] = []
    for step in steps:
        updated_step = dict(step)
        metadata = dict(updated_step.get("metadata") or {})
        metadata.update(
            {
                "trajectory_accuracy_reward": accuracy_reward,
                "trajectory_total_api_cost": total_api_cost,
                "trajectory_api_cost_coef": api_cost_coef,
                "trajectory_composite_reward": composite_reward,
            }
        )
        updated_step["metadata"] = metadata
        annotated_steps.append(updated_step)
    return annotated_steps


def _convert_one_tree(
    tree_path: Path,
    source_tasks: dict[str, dict[str, Any]],
    *,
    trajectory_mode: str,
    incorrect_reward: float,
    threshold: float,
    api_cost_coef: float,
) -> list[dict[str, Any]]:
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    task_id = str(tree.get("task_id", tree_path.parent.name))
    task = copy.deepcopy(source_tasks.get(task_id) or _fallback_task(task_id, tree))

    node_map = {str(node.get("node_id")): node for node in tree.get("nodes", [])}
    rows: list[dict[str, Any]] = []
    for trajectory in _select_trajectories(tree, trajectory_mode):
        node_ids = list(trajectory.get("node_ids", []))
        if not node_ids:
            continue

        steps = _reconstruct_main_steps(
            task,
            node_ids,
            node_map,
            threshold=threshold,
        )
        if not steps:
            continue

        accuracy_reward = _trajectory_accuracy_reward(trajectory, incorrect_reward)
        total_api_cost = _trajectory_total_api_cost(node_ids, node_map)
        composite_reward = accuracy_reward - (api_cost_coef * total_api_cost)
        annotated_steps = _annotate_steps_with_reward_metadata(
            steps,
            accuracy_reward=accuracy_reward,
            total_api_cost=total_api_cost,
            api_cost_coef=api_cost_coef,
            composite_reward=composite_reward,
        )

        rows.append(
            {
                "task_id": task_id,
                "data_source": str(task.get("data_source", "mas_orchestra_tree_search")),
                "is_correct": bool(trajectory.get("correct")) if trajectory.get("correct") is not None else None,
                "trajectory_reward": composite_reward,
                "steps": annotated_steps,
            }
        )
    return rows


def convert_samples_dir(
    samples_dir: Path,
    *,
    source_dataset: str,
    trajectory_mode: str,
    incorrect_reward: float,
    threshold: float,
    api_cost_coef: float,
) -> list[dict[str, Any]]:
    source_tasks = _load_source_tasks(source_dataset)
    rows: list[dict[str, Any]] = []
    for tree_path in sorted(samples_dir.glob("*/tree.json")):
        rows.extend(
            _convert_one_tree(
                tree_path,
                source_tasks,
                trajectory_mode=trajectory_mode,
                incorrect_reward=incorrect_reward,
                threshold=threshold,
                api_cost_coef=api_cost_coef,
            )
        )
    return validate_offline_replay_rows(rows, source_path=samples_dir)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MAS Orchestra tree-search sample logs into offline replay data with reward = acc - coef * trajectory_api_cost."
    )
    parser.add_argument("--samples-dir", required=True, help="Directory containing per-task sample subdirectories with tree.json files.")
    parser.add_argument("--output", default=None, help="Output path. Defaults to <samples_dir>/../mas_orchestra_offline_replay.arrow")
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET, help="Registered source dataset used to recover multimodal task payloads.")
    parser.add_argument(
        "--trajectory-mode",
        choices=("all", "correct", "best", "best_correct"),
        default="correct",
        help="Which final trajectories to export per task.",
    )
    parser.add_argument(
        "--incorrect-reward",
        type=float,
        default=0.0,
        help="Accuracy reward assigned to incorrect trajectories.",
    )
    parser.add_argument(
        "--submit-threshold",
        type=float,
        default=0.75,
        help="Threshold used when reconstructing main prompts.",
    )
    parser.add_argument(
        "--api-cost-coef",
        type=float,
        default=0.1,
        help="Coefficient in composite reward: accuracy_reward - api_cost_coef * total_api_cost.",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir).expanduser().resolve()
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    output_path = Path(args.output).expanduser().resolve() if args.output else samples_dir.parent / DEFAULT_OUTPUT_NAME

    rows = convert_samples_dir(
        samples_dir,
        source_dataset=args.source_dataset,
        trajectory_mode=args.trajectory_mode,
        incorrect_reward=args.incorrect_reward,
        threshold=args.submit_threshold,
        api_cost_coef=args.api_cost_coef,
    )
    if not rows:
        raise RuntimeError("No offline replay rows were produced. Check trajectory-mode and source dataset.")

    write_output(rows, output_path)

    num_steps = sum(_safe_len(row.get("steps", [])) for row in rows)
    num_correct = sum(1 for row in rows if row.get("is_correct"))
    composite_rewards = [float(row.get("trajectory_reward", 0.0) or 0.0) for row in rows]
    accuracy_rewards = [
        float((row.get("steps") or [{}])[0].get("metadata", {}).get("trajectory_accuracy_reward", 0.0) or 0.0)
        for row in rows
    ]
    total_api_costs = [
        float((row.get("steps") or [{}])[0].get("metadata", {}).get("trajectory_total_api_cost", 0.0) or 0.0)
        for row in rows
    ]

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Correct rows: {num_correct}")
    print(f"Average steps per row: {num_steps / len(rows):.2f}")
    print(f"Average accuracy reward: {_mean(accuracy_rewards):.6f}")
    print(f"Average trajectory API cost: {_mean(total_api_costs):.6f}")
    print(f"Average composite reward: {_mean(composite_rewards):.6f}")


if __name__ == "__main__":
    main()
