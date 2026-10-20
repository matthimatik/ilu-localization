from core.fusion import fuse
from orchestrator.stage import StageContext
from orchestrator.stages.util import load_model
from schemas.association import Associations
from schemas.identification import IdentifiedILUs
from schemas.image import ImageRef
from schemas.vision import Embeddings, PoseBoxes, TextSpots


class FuseStage:
    name = "fuse"

    def run(self, ctx: StageContext) -> None:
        associations = load_model(ctx, "associations", Associations)
        text_spots = load_model(ctx, "text_spots", TextSpots)
        pose_boxes = load_model(ctx, "pose_boxes", PoseBoxes)
        embeddings = load_model(ctx, "embeddings", Embeddings)
        image_ref = load_model(ctx, "image_ref", ImageRef)
        identified = (
            load_model(ctx, "ilu_ids", IdentifiedILUs)
            if "ilu_ids" in ctx.inputs
            else None
        )
        records = fuse(
            associations,
            text_spots,
            pose_boxes,
            embeddings,
            image_path=image_ref.path,
            identified=identified,
        )
        ctx.store.write_model(ctx.stage_name, "ilu_records", records)
