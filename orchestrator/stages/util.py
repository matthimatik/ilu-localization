from pathlib import Path

from orchestrator.stage import StageContext


def resolve_path(ctx: StageContext, key: str) -> Path:
    path = Path(ctx.cfg[key])
    if path.is_absolute():
        return path
    return Path(ctx.cfg.get("_base_dir", ".")) / path


def load_model(ctx: StageContext, logical: str, model_cls):
    return ctx.store.read_model_path(ctx.inputs[logical], model_cls)
