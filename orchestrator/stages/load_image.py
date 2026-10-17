from orchestrator.stage import StageContext
from orchestrator.stages.util import resolve_path
from schemas.image import ImageRef


class LoadImageStage:
    name = "load_image"

    def run(self, ctx: StageContext) -> None:
        path_str = ctx.cfg["image"]
        resolved = resolve_path(ctx, "image")
        if not resolved.exists():
            raise FileNotFoundError(f"image not found: {resolved}")
        ref = ImageRef.model_validate(
            {"path": path_str, "camera": ctx.cfg.get("camera")}
        )
        ctx.store.write_model(ctx.stage_name, "image_ref", ref)
