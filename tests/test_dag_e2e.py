from pathlib import Path

from orchestrator.config import build_pipeline
from orchestrator.store import ArtifactStore
from schemas.ilu import ILURecords
from schemas.run import StageStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "golden" / "ilu_records.json"


def test_fixture_pipeline_end_to_end(tmp_path):
    pipeline = build_pipeline(
        REPO_ROOT / "configs" / "pipeline.yaml", base_dir=REPO_ROOT
    )
    store = ArtifactStore(tmp_path / "runs", "test-run")
    manifest = pipeline.run(store)

    assert manifest.status == "success"
    assert all(s.status == StageStatus.success for s in manifest.stages)
    assert [s.stage for s in manifest.stages] == [
        "load_image",
        "textspots",
        "poseboxes",
        "embeddings",
        "associate",
        "identify",
        "fuse",
    ]

    records = store.read_model("fuse", "ilu_records", ILURecords)
    assert len(records.records) == 2
    by_id = {r.ilu_id: r for r in records.records}

    box_a = by_id["box-A"]
    assert box_a.iso_code == "BMOU2372713"
    assert box_a.id_scheme == "iso6346"
    assert box_a.identification_method.value == "code"
    assert box_a.category == "container"
    assert box_a.size_class == "40"
    assert box_a.size_type == "45G1"

    box_b = by_id["box-B"]
    assert box_b.iso_code is None
    assert box_b.identification_method.value == "embedding_rank"
    assert box_b.category == "trailer"

    assert records.image_path == "tests/fixtures/image.png"


def test_fixture_pipeline_matches_golden(tmp_path):
    pipeline = build_pipeline(
        REPO_ROOT / "configs" / "pipeline.yaml", base_dir=REPO_ROOT
    )
    store = ArtifactStore(tmp_path / "runs", "golden-run")
    manifest = pipeline.run(store)
    assert manifest.status == "success"

    actual = store.read_model("fuse", "ilu_records", ILURecords)
    golden = ILURecords.model_validate_json(GOLDEN.read_text())
    assert actual.model_dump() == golden.model_dump()


def test_failed_stage_stops_pipeline(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
pipeline: bad
stages:
  - name: load_image
    impl: load_image
    cfg:
      image: does/not/exist.png
"""
    )
    pipeline = build_pipeline(config, base_dir=REPO_ROOT)
    store = ArtifactStore(tmp_path / "runs", "bad-run")
    manifest = pipeline.run(store)
    assert manifest.status == "failed"
    assert manifest.stages[0].status == StageStatus.failed
    assert "FileNotFoundError" in manifest.stages[0].error
