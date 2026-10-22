from pathlib import Path

from orchestrator.stage import StageContext


def resolve_path(ctx: StageContext, key: str) -> Path:
    path = Path(ctx.cfg[key])
    if path.is_absolute():
        return path
    return Path(ctx.cfg.get("_base_dir", ".")) / path


def resolve_weights(cfg: dict, default: str) -> Path:
    """Resolve the `weights` cfg key against `backend_dir` (itself against `_base_dir`)."""
    weights = Path(cfg.get("weights", default))
    if weights.is_absolute():
        return weights
    backend = Path(cfg["backend_dir"])
    if not backend.is_absolute():
        backend = Path(cfg.get("_base_dir", ".")) / backend
    return backend / weights


def load_model(ctx: StageContext, logical: str, model_cls):
    return ctx.store.read_model_path(ctx.inputs[logical], model_cls)
