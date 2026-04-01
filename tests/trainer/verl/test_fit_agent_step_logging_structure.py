import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_module(relative_path: str) -> ast.Module:
    source = (REPO_ROOT / relative_path).read_text()
    return ast.parse(source)


def _get_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} not found")


def _get_method(class_def: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {method_name} not found in {class_def.name}")


def _is_batch_loop(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "batch_dict"
        and isinstance(node.iter, ast.Attribute)
        and isinstance(node.iter.value, ast.Name)
        and node.iter.value.id == "self"
        and node.iter.attr == "train_dataloader"
    )


def _find_batch_loop(method: ast.FunctionDef) -> ast.For:
    for node in ast.walk(method):
        if _is_batch_loop(node):
            return node
    raise AssertionError("batch loop over self.train_dataloader not found")


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_self_attr(node: ast.AST, attr: str) -> bool:
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr == attr


def _contains_metrics_log(batch_loop: ast.For) -> bool:
    for node in ast.walk(batch_loop):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and _is_name(node.func.value, "logger") and node.func.attr == "log"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "data" and _is_name(keyword.value, "metrics"):
                return True
    return False


def _contains_progress_advance(batch_loop: ast.For) -> bool:
    for node in ast.walk(batch_loop):
        if isinstance(node, ast.Call) and _is_name(node.func, "advance_training_progress"):
            return True
    return False


def _contains_global_step_increment(batch_loop: ast.For) -> bool:
    for node in ast.walk(batch_loop):
        if (
            isinstance(node, ast.AugAssign)
            and _is_self_attr(node.target, "global_steps")
            and isinstance(node.op, ast.Add)
            and isinstance(node.value, ast.Constant)
            and node.value.value == 1
        ):
            return True
    return False


def _contains_stop_profiling(batch_loop: ast.For) -> bool:
    for node in ast.walk(batch_loop):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _is_name(node.func.value, "self") and node.func.attr == "_stop_profiling":
            return True
    return False


@pytest.mark.parametrize(
    ("relative_path", "class_name", "expects_stop_profiling"),
    [
        ("rllm/trainer/verl/agent_ppo_trainer.py", "AgentPPOTrainer", False),
        ("rllm/trainer/verl/agent_workflow_trainer.py", "AgentWorkflowPPOTrainer", True),
        ("rllm/trainer/verl/agent_sdk_trainer.py", "AgentSdkTrainer", True),
    ],
)
def test_fit_agent_logs_and_advances_inside_batch_loop(relative_path: str, class_name: str, expects_stop_profiling: bool):
    module = _parse_module(relative_path)
    class_def = _get_class(module, class_name)
    fit_agent = _get_method(class_def, "fit_agent")
    batch_loop = _find_batch_loop(fit_agent)

    assert _contains_metrics_log(batch_loop), f"{relative_path} should log metrics inside the batch loop"
    assert _contains_progress_advance(batch_loop), f"{relative_path} should update tqdm progress inside the batch loop"
    assert _contains_global_step_increment(batch_loop), f"{relative_path} should increment global_steps inside the batch loop"

    if expects_stop_profiling:
        assert _contains_stop_profiling(batch_loop), f"{relative_path} should stop profiling inside the batch loop"
