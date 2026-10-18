from orchestrator.stage import Stage

from orchestrator.stages.associate import AssociateStage
from orchestrator.stages.fixture import (
    FixtureEmbeddingsStage,
    FixturePoseBoxesStage,
    FixtureTextSpotsStage,
)
from orchestrator.stages.fuse import FuseStage
from orchestrator.stages.instanceseg import InstancesegBackendStage
from orchestrator.stages.load_image import LoadImageStage
from orchestrator.stages.mmocr_backend import MmocrBackendStage
from orchestrator.stages.pose import PoseBackendStage

_REGISTRY: dict[str, type] = {
    "load_image": LoadImageStage,
    "fixture_textspots": FixtureTextSpotsStage,
    "fixture_poseboxes": FixturePoseBoxesStage,
    "fixture_embeddings": FixtureEmbeddingsStage,
    "pose": PoseBackendStage,
    "mmocr": MmocrBackendStage,
    "instanceseg": InstancesegBackendStage,
    "associate": AssociateStage,
    "fuse": FuseStage,
}


def create_stage(impl: str) -> Stage:
    try:
        return _REGISTRY[impl]()
    except KeyError:
        raise ValueError(
            f"unknown stage impl {impl!r}; known: {sorted(_REGISTRY)}"
        ) from None
