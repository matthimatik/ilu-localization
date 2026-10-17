import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.visualization import (
    build_overlay,
    build_report,
    find_latest_run,
    load_run,
    plot_scene,
    write_html,
)
from core.visualization.draw import (
    _overlaps,
    plan_labels,
    record_label_items,
    record_table_cells,
    stroke,
)
from core.visualization.report import write_report
from core.visualization.scene import spot_world
from orchestrator.config import build_pipeline
from orchestrator.store import ArtifactStore
from schemas.association import Association, Associations
from schemas.ilu import IdentificationMethod, ILURecord, ILURecords
from schemas.vision import TextSpots
from tests.builders import make_box, make_spot

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def fixture_run(tmp_path):
    pipeline = build_pipeline(REPO_ROOT / "configs" / "pipeline.yaml", base_dir=REPO_ROOT)
    store = ArtifactStore(tmp_path / "runs", "viz-run")
    manifest = pipeline.run(store)
    assert manifest.status == "success"
    return store.run_dir


def test_overlay_adds_annotation_and_table(fixture_run):
    artifacts = load_run(fixture_run, base_dir=REPO_ROOT)
    image = cv2.imread(str(artifacts.image_path))
    assert image is not None

    overlay = build_overlay(
        image,
        artifacts.records,
        text_spots=artifacts.text_spots,
        associations=artifacts.associations,
        title="test title",
    )

    assert overlay.shape[1] == image.shape[1]
    assert overlay.shape[0] > image.shape[0]  # table strip appended below
    # boxes, badges and OCR quads painted over the source pixels
    assert (overlay[: image.shape[0]] != image).any()
    # table strip starts with its header band (row background follows below)
    assert (overlay[image.shape[0]] == 48).all()


def test_labels_default_to_id_badges():
    records = [
        ILURecord(
            ilu_id="box-A",
            category="container",
            iso_code="BMOU2372713",
            size_class="40",
            identification_method=IdentificationMethod.code,
        ),
        ILURecord(
            ilu_id="box-B",
            category="trailer",
            identification_method=IdentificationMethod.embedding_rank,
        ),
    ]

    items = record_label_items(records)

    assert all(item.badge for item in items)
    assert [item.text for item in items] == ["#box-A", "#box-B"]
    assert all(item.sub is None for item in items)


def test_scene_from_fixture_run(fixture_run, tmp_path):
    artifacts = load_run(fixture_run, base_dir=REPO_ROOT)
    fig = plot_scene(
        artifacts.records, artifacts.world_by_id, artifacts.camera, title="scene"
    )
    assert fig is not None
    # fixture run has no pose backend output: centre markers only, no camera
    assert len(fig.data) == len(artifacts.records.records)
    assert all(trace.type == "scatter3d" for trace in fig.data)

    path = write_html(fig, tmp_path / "scene.html")
    assert "scatter3d" in path.read_text()


def test_scene_hover_includes_ocr(fixture_run):
    artifacts = load_run(fixture_run, base_dir=REPO_ROOT)
    fig = plot_scene(
        artifacts.records,
        artifacts.world_by_id,
        artifacts.camera,
        text_spots=artifacts.text_spots,
        associations=artifacts.associations,
    )
    assert fig is not None
    hover_parts: list[str] = []
    for trace in fig.data:
        hover = trace.hovertext
        if hover is None:
            continue
        hover_parts.append(hover if isinstance(hover, str) else " ".join(hover))
    hovers = " ".join(hover_parts)
    # box-A is associated with the BMOU code and the 45G1 size-code spots
    assert "OCR: " in hovers
    assert "BMOU2372713 0.94" in hovers


def test_spot_world_lands_on_box():
    # camera at the origin looking +z; a 1x1x1 box from z=2..3 centred on the axis;
    # a spot at the image centre must hit the box's near face at z=2
    camera = {
        "r_cw": np.eye(3).tolist(),
        "cam": [0.0, 0.0, 0.0],
        "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        "image_size": [100, 100],
    }
    corners = np.array(
        [
            (-0.5, -0.5, 2.0), (-0.5, -0.5, 3.0),
            (-0.5, 0.5, 3.0), (-0.5, 0.5, 2.0),
            (0.5, -0.5, 2.0), (0.5, -0.5, 3.0),
            (0.5, 0.5, 3.0), (0.5, 0.5, 2.0),
        ]
    )

    hit = spot_world(camera, (50.0, 50.0), corners)

    assert hit is not None
    assert np.allclose(hit, (0.0, 0.0, 2.0), atol=1e-6)


def test_labels_avoid_ocr_quads():
    # the badge's default spot (above the box top) sits inside an OCR quad rect
    records = [
        ILURecord(
            ilu_id="box-A",
            category="container",
            size_class="40",
            identification_method=IdentificationMethod.code,
            pose=make_box(box_id="box-A", x0=100, y0=100, x1=610, y1=500),
        )
    ]
    lw, fs, tf = stroke(np.zeros((700, 800, 3), np.uint8))
    obstacle = (100, 30, 620, 95)

    placements = plan_labels(
        800, 700, record_label_items(records), "title", lw, fs, tf,
        obstacles=[obstacle],
    )

    assert len(placements) == 1
    assert not _overlaps(placements[0].rect, [obstacle])


def test_scene_iso_tags_and_spot_markers():
    # synthetic geometry: camera at the origin, box at z=2..3, spot at image centre
    record = ILURecord(
        ilu_id="box-A",
        category="container",
        size_class="40",
        iso_code="BMOU2372713",
        identification_method=IdentificationMethod.code,
        pose=make_box(box_id="box-A"),
    )
    records = ILURecords(image_path="img.png", records=[record])
    corners = np.array(
        [
            (-0.5, -0.5, 2.0), (-0.5, -0.5, 3.0),
            (-0.5, 0.5, 3.0), (-0.5, 0.5, 2.0),
            (0.5, -0.5, 2.0), (0.5, -0.5, 3.0),
            (0.5, 0.5, 3.0), (0.5, 0.5, 2.0),
        ]
    )
    camera = {
        "r_cw": np.eye(3).tolist(),
        "cam": [0.0, 0.0, 0.0],
        "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        "image_size": [100, 100],
    }
    spots = TextSpots(image_path="img.png", spots=[make_spot(center=(50.0, 50.0))])
    associations = Associations(
        associations=[Association(pose_box_id="box-A", text_spot_indices=[0])]
    )

    fig = plot_scene(
        records,
        {"box-A": corners},
        camera,
        text_spots=spots,
        associations=associations,
    )

    assert fig is not None
    by_name = {trace.name: trace for trace in fig.data}
    tag = by_name["iso codes"]
    assert list(tag.text) == ["BMOU2372713"]
    assert np.allclose([tag.x[0], tag.y[0], tag.z[0]], (0.0, 0.0, 3.0))  # top centre
    marker = by_name["text spots"]
    assert len(marker.x) == 1
    assert np.allclose(  # ray through the quad centre hits the near face
        [marker.x[0], marker.y[0], marker.z[0]], (0.0, 0.0, 2.0)
    )
    assert "OCR: BMOU2372713 0.90" in by_name["code"].hovertext


def test_record_table_lists_details(fixture_run):
    artifacts = load_run(fixture_run, base_dir=REPO_ROOT)

    headers, rows = record_table_cells(
        artifacts.records.records, artifacts.text_spots, artifacts.associations
    )

    assert headers[0] == "id" and "OCR texts" in headers
    by_id = {row[0]: row for row in rows}
    box_a = by_id["box-A"]
    assert box_a[3] == "BMOU2372713"  # ISO column
    assert box_a[4] == "code"  # ident method
    assert box_a[9] == "2"  # associated spots
    assert "BMOU2372713 0.94" in box_a[10] and "45G1 0.87" in box_a[10]
    # box-B has no text associations
    assert by_id["box-B"][9] == "0" and by_id["box-B"][10] == "—"


def test_report_lists_records(fixture_run, tmp_path):
    artifacts = load_run(fixture_run, base_dir=REPO_ROOT)
    html = build_report(
        artifacts.records,
        overlay_jpeg=b"\xff\xd8\xff\xd9",
        links={"3D scene": "scene.html"},
        associations=artifacts.associations,
        text_spots=artifacts.text_spots,
        title="report title",
    )
    assert "box-A" in html and "box-B" in html
    assert "BMOU2372713" in html
    assert "embedding_rank" in html
    assert "OCR texts" in html
    assert "<table>" in html
    assert "data:image/jpeg;base64" in html
    assert 'href="scene.html"' in html

    path = write_report(html, tmp_path / "report.html")
    assert path.exists()


def test_label_placement_never_overlaps():
    # stacked boxes compete for the same spot; the planner must separate them
    records = [
        ILURecord(
            ilu_id=f"box-{i}",
            category="container",
            size_class="20",
            identification_method=IdentificationMethod.unknown,
            pose=make_box(box_id=f"box-{i}", x0=1200, y0=800, x1=1700, y1=1300),
            confidences={"pose": 0.9},
        )
        for i in range(8)
    ]
    lw, fs, tf = stroke(np.zeros((2160, 3840, 3), np.uint8))

    # worst case: verbose full labels
    placements = plan_labels(
        3840, 2160, record_label_items(records, "all"), "title", lw, fs, tf
    )

    rects = [placement.rect for placement in placements]
    assert len(rects) == len(records)
    for i, rect in enumerate(rects):
        for other in rects[i + 1 :]:
            assert not _overlaps(rect, [other])


def test_find_latest_run_picks_newest_successful(tmp_path):
    runs = tmp_path / "runs"
    for run_id, status in (("20260101-000000-a", "success"), ("20260102-000000-b", "failed")):
        run_dir = runs / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"status": status}))

    assert find_latest_run(runs).name == "20260101-000000-a"

    (runs / "20260101-000000-a" / "manifest.json").write_text(
        json.dumps({"status": "failed"})
    )
    with pytest.raises(FileNotFoundError):
        find_latest_run(runs)


def test_visualize_run_cli(fixture_run, tmp_path):
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "visualize_run.py"),
            "--runs-root",
            str(fixture_run.parent),
            "--run-id",
            fixture_run.name,
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "overlay.jpg").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "scene.html").exists()
    assert (out_dir / "overlay.jpg").stat().st_size > 0
