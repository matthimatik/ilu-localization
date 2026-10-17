from orchestrator.stage import StageContext
from orchestrator.stages.util import load_model, resolve_path
from schemas.image import ImageRef
from schemas.vision import Embeddings, PoseBoxes, TextSpots


class FixtureTextSpotsStage:
    name = "fixture_textspots"

    def run(self, ctx: StageContext) -> None:
        image_ref = load_model(ctx, "image", ImageRef)
        spots = TextSpots.model_validate_json(resolve_path(ctx, "source").read_text())
        spots.image_path = image_ref.path
        ctx.store.write_model(ctx.stage_name, "text_spots", spots)


class FixturePoseBoxesStage:
    name = "fixture_poseboxes"

    def run(self, ctx: StageContext) -> None:
        image_ref = load_model(ctx, "image", ImageRef)
        boxes = PoseBoxes.model_validate_json(resolve_path(ctx, "source").read_text())
        boxes.image_path = image_ref.path
        ctx.store.write_model(ctx.stage_name, "pose_boxes", boxes)


class FixtureEmbeddingsStage:
    name = "fixture_embeddings"

    def run(self, ctx: StageContext) -> None:
        embeddings = Embeddings.model_validate_json(
            resolve_path(ctx, "source").read_text()
        )
        ctx.store.write_model(ctx.stage_name, "embeddings", embeddings)
