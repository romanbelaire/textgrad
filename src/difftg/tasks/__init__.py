from ..config import TaskConfig
from .toy import ToyTask, build_toy_tasks


def build_tasks(cfg: TaskConfig):
    if cfg.kind == "toy":
        return build_toy_tasks(cfg)
    if cfg.kind == "gsm8k":
        from .gsm8k import build_gsm8k_tasks
        return build_gsm8k_tasks(cfg)
    raise ValueError(f"Unknown task kind: {cfg.kind!r}")
