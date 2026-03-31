from __future__ import annotations

from tqdm.auto import tqdm


def create_training_progress_bar(trainer, *, desc: str = "Training"):
    total_steps = getattr(trainer, "total_training_steps", None)
    if total_steps is not None and total_steps > 0:
        total = max(int(total_steps) - 1, 0)
    else:
        try:
            total = len(trainer.train_dataloader) * int(trainer.config.trainer.total_epochs)
        except TypeError:
            total = None

    initial = max(int(getattr(trainer, "global_steps", 0)) - 1, 0)
    if total is not None:
        initial = min(initial, total)

    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        dynamic_ncols=True,
        smoothing=0.1,
    )


def advance_training_progress(progress_bar, *, epoch: int, global_step: int) -> None:
    progress_bar.set_postfix(epoch=epoch + 1, step=global_step, refresh=False)
    progress_bar.update(1)
