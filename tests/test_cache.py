from pathlib import Path

from orchestrator.cache import StageCache
from orchestrator.dag import Pipeline
from orchestrator.stage import StageSpec
from orchestrator.stages.instanceseg import InstancesegBackendStage
from orchestrator.stages.pose import PoseBackendStage
from orchestrator.store import ArtifactStore
from schemas.run import StageStatus


def _pose_cfg(tmp_path: Path) -> dict:
    backend = tmp_path / "backend"
    (backend / "models").mkdir(parents=True)
    return {"backend_dir": str(backend), "_base_dir": str(tmp_path)}


def test_pose_stage_cache_files_resolves_default_weights(tmp_path):
    cfg = _pose_cfg(tmp_path)
    weights = Path(cfg["backend_dir"]) / "models" / "container-corners-yolov8m-1280.pt"
    assert PoseBackendStage().cache_files(cfg) == {"weights": weights}


def test_pose_stage_cache_files_resolves_cfg_override(tmp_path):
    cfg = _pose_cfg(tmp_path)
    cfg["weights"] = "models/custom.pt"
    weights = Path(cfg["backend_dir"]) / "models" / "custom.pt"
    assert PoseBackendStage().cache_files(cfg) == {"weights": weights}


def test_instanceseg_stage_cache_files_resolves_weights(tmp_path):
    cfg = _pose_cfg(tmp_path)
    weights = Path(cfg["backend_dir"]) / "models" / "ilu-seg-yolov8m-1024.pt"
    assert InstancesegBackendStage().cache_files(cfg) == {"weights": weights}


def test_dag_cache_invalidates_when_declared_file_changes(tmp_path):
    calls: list[int] = []

    class DummyStage:
        name = "pose"

        def cache_files(self, cfg: dict) -> dict[str, Path]:
            return {"weights": cfg["_weights"]}

        def run(self, ctx) -> None:
            calls.append(1)
            (ctx.store.stage_dir(ctx.stage_name) / "out.json").write_text("{}")

    weights = tmp_path / "w.pt"
    weights.write_bytes(b"v1")
    pipeline = Pipeline("t", [StageSpec(name="pose", stage=DummyStage(), cfg={"_weights": weights})])
    cache = StageCache(tmp_path / "cache")

    m1 = pipeline.run(ArtifactStore(tmp_path / "runs", "r1"), cache)
    assert m1.status == "success" and len(calls) == 1

    m2 = pipeline.run(ArtifactStore(tmp_path / "runs", "r2"), cache)
    assert m2.stages[0].status == StageStatus.cached and len(calls) == 1

    weights.write_bytes(b"v2")  # same path, new content -> cache miss
    m3 = pipeline.run(ArtifactStore(tmp_path / "runs", "r3"), cache)
    assert m3.stages[0].status == StageStatus.success and len(calls) == 2
