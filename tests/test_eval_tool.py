import json
import subprocess
import sys
from pathlib import Path

from core.evaluation.ground_truth import parse_iso_from_label
from schemas.evaluation import EvalReport
from schemas.ilu import (
    IdentificationMethod,
    ILURecord,
    ILURecords,
)
from schemas.vision import Corner, PoseBox

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_TOOL = REPO_ROOT / "tools" / "eval_run.py"


def _world_box(cx: float = 0.0) -> list[list[float]]:
    length, width, height = 6.0, 2.44, 2.6
    x0, x1 = cx - length / 2, cx + length / 2
    y0, y1 = -width / 2, width / 2
    bottom = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
    top = [(x, y, height) for x, y, _ in bottom]
    return [[float(v) for v in c] for c in bottom + top]


def _uv(offset: float = 0.0) -> list[list[float]]:
    return [[100.0 + offset + 10 * i, 200.0 + offset + 5 * i] for i in range(8)]


def _write_run(run_dir: Path) -> None:
    image_path = run_dir / "image.jpg"
    image_path.write_bytes(b"fake-image")

    det0_world = [[v + 0.05 for v in c] for c in _world_box(0.0)]
    det1_world = [[v + 0.05 for v in c] for c in _world_box(10.0)]

    raw_dir = run_dir / "pose" / "raw" / "rgb"
    raw_dir.mkdir(parents=True)
    (raw_dir / "detections.json").write_text(
        json.dumps(
            {
                "image": "rgb.jpg",
                "size": [3840, 2160],
                "detections": [
                    {"cls": "container", "conf": 0.9, "uv": _uv(0.0), "type_hint": "20"},
                    {"cls": "container", "conf": 0.8, "uv": _uv(300.0), "type_hint": "20"},
                ],
            }
        )
    )
    (raw_dir / "boxes.json").write_text(
        json.dumps(
            {
                "image": "rgb.jpg",
                "method": "pnp",
                "camera": {
                    "r_cw": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "cam": [0.0, 0.0, 10.0],
                    "K": [[5000, 0, 1920], [0, 5000, 1080], [0, 0, 1]],
                    "image_size": [3840, 2160],
                },
                "boxes": [
                    {
                        "cls": "container",
                        "size": "20",
                        "n_corners": 8,
                        "rms": 0.2,
                        "uv": _uv(),
                        "visible": [True] * 8,
                        "world": det0_world,
                    },
                    {
                        "cls": "container",
                        "size": "20",
                        "n_corners": 8,
                        "rms": 0.2,
                        "uv": _uv(300.0),
                        "visible": [True] * 8,
                        "world": det1_world,
                    },
                ],
            }
        )
    )

    corners_px = [Corner(position_px=(float(i), float(i)), confidence=0.9) for i in range(8)]
    records = ILURecords(
        image_path=str(image_path),
        records=[
            ILURecord(
                ilu_id="det-0",
                category="container",
                iso_code="KLJU2372713",
                size_class="20",
                identification_method=IdentificationMethod.code,
                pose=PoseBox(
                    box_id="det-0",
                    corners_px=corners_px,
                    size_class="20",
                    method="pnp",
                    confidence=0.9,
                ),
                confidences={"pose": 0.9},
            ),
            ILURecord(
                ilu_id="det-1",
                category="container",
                iso_code="IGQU5750759",  # right owner+serial, wrong check digit
                size_class="20",
                identification_method=IdentificationMethod.code,
                pose=PoseBox(
                    box_id="det-1",
                    corners_px=corners_px,
                    size_class="20",
                    method="pnp",
                    confidence=0.8,
                ),
                confidences={"pose": 0.8},
            ),
        ],
    )
    fuse_dir = run_dir / "fuse"
    fuse_dir.mkdir()
    (fuse_dir / "ilu_records.json").write_text(records.model_dump_json(indent=2))


def _write_gt(tmp_path: Path) -> Path:
    gt_path = tmp_path / "scene" / "pose.jsonl"
    gt_path.parent.mkdir(parents=True)
    labels = ["KLJU237271 3", "IGQU575075 2"]
    lines = []
    for cx, label in zip([0.0, 10.0], labels):
        entries = [
            {
                "world_position": [i + 1, corner],
                "image_position": [100 + i, 200],
            }
            for i, corner in enumerate(_world_box(cx))
        ]
        lines.append(json.dumps({"corners": entries, "model": "20", "label": label}))
    gt_path.write_text("\n".join(lines))
    return gt_path


def test_eval_tool_end_to_end(tmp_path):
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    _write_run(run_dir)
    gt_path = _write_gt(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(EVAL_TOOL),
            "--runs-root",
            str(tmp_path / "runs"),
            "--run-id",
            "test-run",
            "--gt",
            str(gt_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    report_path = run_dir / "eval" / "eval_report.json"
    assert report_path.exists()
    report = EvalReport.model_validate_json(report_path.read_text())
    assert report.run_id == "test-run"
    assert report.metrics.n_matched == 2
    assert report.metrics.recall == 1.0
    assert report.metrics.precision == 1.0
    assert report.metrics.id_relaxed_accuracy == 1.0
    assert report.metrics.id_accuracy == 0.5  # det-1 has a wrong check digit
    by_id = {p.pred_ilu_id: p for p in report.pairs}
    assert by_id["det-0"].id_correct is True
    assert by_id["det-1"].id_correct is False
    assert by_id["det-1"].id_relaxed_correct is True
    assert "precision" in result.stdout


def test_eval_tool_requires_world_corners(tmp_path):
    run_dir = tmp_path / "runs" / "no-pnp"
    run_dir.mkdir(parents=True)
    image_path = run_dir / "image.jpg"
    image_path.write_bytes(b"fake-image")
    records = ILURecords(
        image_path=str(image_path),
        records=[
            ILURecord(
                ilu_id="det-0",
                category="container",
                identification_method=IdentificationMethod.unknown,
                confidences={},
            )
        ],
    )
    fuse_dir = run_dir / "fuse"
    fuse_dir.mkdir()
    (fuse_dir / "ilu_records.json").write_text(records.model_dump_json(indent=2))
    gt_path = _write_gt(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(EVAL_TOOL),
            "--runs-root",
            str(tmp_path / "runs"),
            "--run-id",
            "no-pnp",
            "--gt",
            str(gt_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "no PnP world corners" in (result.stderr + result.stdout)


def test_gt_auto_discovery_next_to_image(tmp_path):
    """pose.jsonl next to the scene image is picked up without --gt."""
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    gt_lines = []
    entries = [
        {
            "world_position": [i + 1, corner],
            "image_position": [100 + i, 200],
        }
        for i, corner in enumerate(_world_box(0.0))
    ]
    gt_lines.append(
        json.dumps(
            {"corners": entries, "model": "20", "label": "KLJU237271 3"}
        )
    )
    (scene_dir / "pose.jsonl").write_text("\n".join(gt_lines))
    image_path = scene_dir / "rgb.jpg"
    image_path.write_bytes(b"fake-image")

    run_dir = tmp_path / "runs" / "auto-gt"
    run_dir.mkdir(parents=True)
    raw_dir = run_dir / "pose" / "raw" / "rgb"
    raw_dir.mkdir(parents=True)
    (raw_dir / "detections.json").write_text(
        json.dumps(
            {
                "image": "rgb.jpg",
                "size": [3840, 2160],
                "detections": [
                    {"cls": "container", "conf": 0.9, "uv": _uv(), "type_hint": "20"}
                ],
            }
        )
    )
    (raw_dir / "boxes.json").write_text(
        json.dumps(
            {
                "image": "rgb.jpg",
                "method": "pnp",
                "boxes": [
                    {
                        "cls": "container",
                        "size": "20",
                        "uv": _uv(),
                        "world": _world_box(0.0),
                    }
                ],
            }
        )
    )
    corners_px = [
        Corner(position_px=(float(i), float(i)), confidence=0.9) for i in range(8)
    ]
    records = ILURecords(
        image_path=str(image_path),
        records=[
            ILURecord(
                ilu_id="det-0",
                category="container",
                iso_code=parse_iso_from_label("KLJU237271 3"),
                size_class="20",
                identification_method=IdentificationMethod.code,
                pose=PoseBox(
                    box_id="det-0",
                    corners_px=corners_px,
                    size_class="20",
                    method="pnp",
                    confidence=0.9,
                ),
                confidences={"pose": 0.9},
            )
        ],
    )
    fuse_dir = run_dir / "fuse"
    fuse_dir.mkdir()
    (fuse_dir / "ilu_records.json").write_text(records.model_dump_json(indent=2))

    result = subprocess.run(
        [
            sys.executable,
            str(EVAL_TOOL),
            "--runs-root",
            str(tmp_path / "runs"),
            "--run-id",
            "auto-gt",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    report = EvalReport.model_validate_json(
        (run_dir / "eval" / "eval_report.json").read_text()
    )
    assert report.metrics.n_matched == 1
    assert report.metrics.recall == 1.0
    assert "pose.jsonl" in report.gt_source
