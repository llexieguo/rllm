from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from examples.mas_orchestra.memory import MainMemory
from examples.mas_orchestra.offline_replay import validate_offline_replay_rows
from examples.mas_orchestra.prompts import build_main_prompt
from examples.mas_orchestra.schema import AttemptRecord, DelegateResult
from examples.mas_orchestra.subclients import task_to_reasoning_sample
from rllm.data.dataset import DatasetRegistry

DEFAULT_SOURCE_DATASET = "sgi_reasoning_mas_orchestra"
DEFAULT_OUTPUT_NAME = "mas_orchestra_offline_replay.arrow"
SYSTEM_PROMPT = "You are a strict orchestration controller. Output JSON only."


def _safe_len(value: Any) -> int:
    return len(value) if hasattr(value, "__len__") else 0


def _load_source_tasks(dataset_name: str) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        dataset = DatasetRegistry.load_dataset(dataset_name, split)
        if dataset is None:
            continue
        for row in dataset.get_data():
            task_id = str(row.get("task_id", row.get("idx", row.get("id", ""))))
            if task_id:
                tasks[task_id] = dict(row)
    return tasks


def _fallback_task(task_id: str, tree: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "question": str(tree.get("question", "")),
        "options": [str(option) for option in tree.get("options", [])],
        "answer_index": 0,
        "steps": [],
        "discipline": str(tree.get("discipline", "unknown")),
        "images": [],
        "data_source": "mas_orchestra_tree_search",
    }


def _select_trajectories(tree: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    trajectories = list(tree.get("trajectories", []))
    if mode == "all":
        return trajectories
    if mode == "correct":
        return [trajectory for trajectory in trajectories if trajectory.get("correct")]
    if mode == "best":
        best_leaf_node_id = tree.get("best_leaf_node_id")
        return [trajectory for trajectory in trajectories if trajectory.get("leaf_node_id") == best_leaf_node_id]
    if mode == "best_correct":
        if not tree.get("best_leaf_correct"):
            return []
        best_leaf_node_id = tree.get("best_leaf_node_id")
        return [trajectory for trajectory in trajectories if trajectory.get("leaf_node_id") == best_leaf_node_id]
    raise ValueError(f"Unsupported trajectory mode: {mode}")


def _trajectory_reward(trajectory: dict[str, Any], reward_mode: str, incorrect_reward: float) -> float:
    if reward_mode == "binary_correct":
        return 1.0 if trajectory.get("correct") else incorrect_reward
    if reward_mode == "confidence":
        confidence = trajectory.get("confidence")
        if confidence is None:
            return incorrect_reward
        return float(confidence)
    if reward_mode == "selection_score":
        score = trajectory.get("selection_score")
        if score is None:
            return incorrect_reward
        return float(score)
    raise ValueError(f"Unsupported reward mode: {reward_mode}")


def _main_messages(task: dict[str, Any], memory: MainMemory, attempt_index: int, sub_models: list[str], threshold: float) -> list[dict[str, Any]]:
    sample = task_to_reasoning_sample(task)
    prompt = build_main_prompt(
        sample=sample,
        attempt_history_text=memory.as_brief_text(),
        attempt_index=attempt_index,
        max_attempts=max(3, attempt_index),
        sub_models=sub_models,
        threshold=threshold,
    )
    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    if sample.images:
        user_message["images"] = copy.deepcopy(sample.images)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        user_message,
    ]


def _build_delegate_result(
    attempt: dict[str, Any],
    node: dict[str, Any],
    *,
    is_last_delegate_in_node: bool,
) -> DelegateResult:
    payload = dict(attempt.get("delegate_result", {}))
    raw_answer_text = node.get("final_answer_text", "") if is_last_delegate_in_node else ""
    reasoning_summary = node.get("reasoning_summary", "") if is_last_delegate_in_node else ""
    return DelegateResult(
        raw_answer_text=str(raw_answer_text or ""),
        boxed_letter=payload.get("boxed_letter"),
        confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
        reasoning_summary=str(reasoning_summary or ""),
        parse_ok=bool(payload.get("parse_ok", False)),
        error=str(payload["error"]) if payload.get("error") is not None else None,
        cost=float(payload.get("cost", 0.0) or 0.0),
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        provider_model=str(attempt.get("selected_model")) if attempt.get("selected_model") is not None else None,
    )


def _reconstruct_main_steps(
    task: dict[str, Any],
    node_ids: Iterable[str],
    node_map: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    memory = MainMemory()
    steps: list[dict[str, Any]] = []

    for node_id in node_ids:
        if node_id == "root":
            continue

        node = node_map[node_id]
        attempts = list(node.get("orchestra_attempts", []))
        delegate_indices = [idx for idx, attempt in enumerate(attempts) if attempt.get("delegate_result")]
        last_delegate_idx = delegate_indices[-1] if delegate_indices else None

        for attempt_idx, attempt in enumerate(attempts):
            attempt_number = int(attempt.get("attempt_index", attempt_idx + 1))
            sampled_pool = [str(model) for model in node.get("model_pool", []) if str(model).strip()]
            selected_model = attempt.get("selected_model")
            if selected_model is not None and str(selected_model).strip() and str(selected_model) not in sampled_pool:
                sampled_pool.append(str(selected_model))

            messages = _main_messages(
                task,
                memory,
                attempt_number,
                sampled_pool,
                threshold,
            )
            raw_response = str(attempt.get("orchestra_raw_response", ""))
            delegate_payload = attempt.get("delegate_result")
            steps.append(
                {
                    "messages": messages,
                    "response": raw_response,
                    "step_reward": 0.0,
                    "trainable": True,
                    "step_type": "main",
                    "model": str(attempt.get("orchestra_model", "local-policy")),
                    "metadata": {
                        "source_node_id": node_id,
                        "source_parent_id": node.get("parent_id"),
                        "node_depth": int(node.get("depth", 0) or 0),
                        "tree_round_index": int(node.get("round_index", 0) or 0),
                        "attempt_index": attempt_number,
                        "action": attempt.get("action"),
                        "selected_model": selected_model,
                        "chosen_model": node.get("chosen_model"),
                        "delegate_boxed_letter": delegate_payload.get("boxed_letter") if isinstance(delegate_payload, dict) else None,
                        "delegate_confidence": delegate_payload.get("confidence") if isinstance(delegate_payload, dict) else None,
                        "delegate_parse_ok": delegate_payload.get("parse_ok") if isinstance(delegate_payload, dict) else None,
                    },
                }
            )

            if delegate_payload:
                delegate_result = _build_delegate_result(
                    attempt,
                    node,
                    is_last_delegate_in_node=attempt_idx == last_delegate_idx,
                )
                memory.add_attempt(
                    AttemptRecord(
                        attempt_index=attempt_number,
                        model=str(selected_model or attempt.get("orchestra_model") or "unknown"),
                        instruction=str(attempt.get("instruction", "") or ""),
                        delegate_result=delegate_result,
                        main_reasoning=str(attempt.get("orchestra_reasoning", "") or ""),
                    )
                )

    return steps


def _convert_one_tree(
    tree_path: Path,
    source_tasks: dict[str, dict[str, Any]],
    *,
    trajectory_mode: str,
    reward_mode: str,
    incorrect_reward: float,
    threshold: float,
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

        rows.append(
            {
                "task_id": task_id,
                "data_source": str(task.get("data_source", "mas_orchestra_tree_search")),
                "is_correct": bool(trajectory.get("correct")) if trajectory.get("correct") is not None else None,
                "trajectory_reward": _trajectory_reward(
                    trajectory,
                    reward_mode,
                    incorrect_reward,
                ),
                "steps": steps,
            }
        )
    return rows


def convert_samples_dir(
    samples_dir: Path,
    *,
    source_dataset: str,
    trajectory_mode: str,
    reward_mode: str,
    incorrect_reward: float,
    threshold: float,
) -> list[dict[str, Any]]:
    source_tasks = _load_source_tasks(source_dataset)
    rows: list[dict[str, Any]] = []
    for tree_path in sorted(samples_dir.glob("*/tree.json")):
        rows.extend(
            _convert_one_tree(
                tree_path,
                source_tasks,
                trajectory_mode=trajectory_mode,
                reward_mode=reward_mode,
                incorrect_reward=incorrect_reward,
                threshold=threshold,
            )
        )
    return validate_offline_replay_rows(rows, source_path=samples_dir)


def write_output(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    if output_path.suffix == ".parquet":
        pq.write_table(table, output_path)
        return
    if output_path.suffix == ".arrow":
        with pa.OSFile(str(output_path), "wb") as sink:
            writer = ipc.new_file(sink, table.schema)
            writer.write_table(table)
            writer.close()
        return
    raise ValueError(f"Unsupported output format: {output_path.suffix}. Use .arrow or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MAS Orchestra tree-search sample logs into offline replay Arrow/Parquet.")
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
        "--reward-mode",
        choices=("binary_correct", "confidence", "selection_score"),
        default="binary_correct",
        help="How to derive trajectory_reward from each final trajectory.",
    )
    parser.add_argument(
        "--incorrect-reward",
        type=float,
        default=0.0,
        help="Reward assigned to incorrect trajectories when reward_mode=binary_correct.",
    )
    parser.add_argument(
        "--submit-threshold",
        type=float,
        default=0.75,
        help="Threshold used when reconstructing main prompts.",
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
        reward_mode=args.reward_mode,
        incorrect_reward=args.incorrect_reward,
        threshold=args.submit_threshold,
    )
    if not rows:
        raise RuntimeError("No offline replay rows were produced. Check trajectory-mode and source dataset.")

    write_output(rows, output_path)

    num_steps = sum(_safe_len(row.get("steps", [])) for row in rows)
    num_correct = sum(1 for row in rows if row.get("is_correct"))
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Correct rows: {num_correct}")
    print(f"Average steps per row: {num_steps / len(rows):.2f}")


if __name__ == "__main__":
    main()
