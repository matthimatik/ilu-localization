from core.association import associate
from orchestrator.stage import StageContext
from orchestrator.stages.util import load_model
from schemas.vision import Embeddings, PoseBoxes, TextSpots


class AssociateStage:
    name = "associate"

    def run(self, ctx: StageContext) -> None:
        text_spots = load_model(ctx, "text_spots", TextSpots)
        pose_boxes = load_model(ctx, "pose_boxes", PoseBoxes)
        embeddings = load_model(ctx, "embeddings", Embeddings)
        margin = float(ctx.cfg.get("margin_px", 0.0))
        result = associate(
            text_spots, pose_boxes, embeddings, margin_px=margin
        )
        ctx.store.write_model(ctx.stage_name, "associations", result)
