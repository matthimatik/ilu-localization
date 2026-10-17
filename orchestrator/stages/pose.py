import json
import shutil
import subprocess
from pathlib import Path

from orchestrator.stage import StageContext
from orchestrator.stages.util import load_model, resolve_path
from schemas.image import ImageRef
from schemas.vision import Corner, PoseBox, PoseBoxes

DEFAULT_WEIGHTS = "models/container-corners-yolov8m-1280.pt"
DEFAULT_TIMEOUT_S = 300.0
UV_MATCH_TOL_PX = 1e-3


def _match_pnp(det: dict, pnp_boxes: list[dict]) -> dict | None:
    det_uv = det["uv"]
    for box in pnp_boxes:
        box_uv = box.get("uv") or []
        if len(box_uv) != len(det_uv):
            continue
        if all(
            abs(box_uv[k][0] - det_uv[k][0]) < UV_MATCH_TOL_PX
            and abs(box_uv[k][1] - det_uv[k][1]) < UV_MATCH_TOL_PX
            for k in range(len(det_uv))
        ):
            return box
    return None


def _center_m(world: list[list[float]]) -> tuple[float, float, float] | None:
    if not world:
        return None
    n = len(world)
    return (
        float(sum(p[0] for p in world) / n),
        float(sum(p[1] for p in world) / n),
        float(sum(p[2] for p in world) / n),
    )


def pose_boxes_from_backend(
    detections: dict,
    boxes_json: dict | None,
    image_path: str,
) -> PoseBoxes:
    pnp_boxes: list[dict] = boxes_json["boxes"] if boxes_json else []
    pnp_method: str = boxes_json["method"] if boxes_json else "yolo"
    built: list[PoseBox] = []
    for index, det in enumerate(detections["detections"]):
        match = _match_pnp(det, pnp_boxes)
        if det.get("type_hint") is not None:
            size_class = str(det["type_hint"])
        elif match is not None:
            size_class = str(match["size"])
        elif det["cls"] == "trailer":
            size_class = "trailer"
        else:
            size_class = "unknown"
        corners = [
            Corner(
                position_px=(
                    None
                    if not det["visible"][k]
                    else (float(det["uv"][k][0]), float(det["uv"][k][1]))
                ),
                confidence=float(det["kpt_conf"][k]),
            )
            for k in range(len(det["uv"]))
        ]
        center = _center_m(match["world"]) if match is not None else None
        built.append(
            PoseBox(
                box_id=f"det-{index}",
                corners_px=corners,
                size_class=size_class,
                center_m=center,
                method=pnp_method if match is not None else "yolo",
                confidence=float(det["conf"]),
            )
        )
    return PoseBoxes(image_path=image_path, boxes=built)


class PoseBackendStage:
    name = "pose"

    def run(self, ctx: StageContext) -> None:
        image_ref = load_model(ctx, "image", ImageRef)
        backend_dir = resolve_path(ctx, "backend_dir")
        if not backend_dir.is_dir():
            raise FileNotFoundError(f"pose backend not found: {backend_dir}")

        image_path = Path(image_ref.path)
        if not image_path.is_absolute():
            image_path = Path(ctx.cfg["_base_dir"]) / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        weights = Path(ctx.cfg.get("weights", DEFAULT_WEIGHTS))
        if not weights.is_absolute():
            weights = backend_dir / weights

        raw_dir = ctx.store.stage_dir(ctx.stage_name) / "raw"
        cmd = [
            "uv",
            "run",
            "pipeline/run.py",
            str(image_path),
            "--out",
            str(raw_dir),
            "--weights",
            str(weights),
        ]
        if "method" in ctx.cfg:
            cmd += ["--method", str(ctx.cfg["method"])]
        for flag, key in (
            ("--conf", "conf"),
            ("--kpt-conf", "kpt_conf"),
            ("--imgsz", "imgsz"),
        ):
            if key in ctx.cfg:
                cmd += [flag, str(ctx.cfg[key])]
        camera = image_ref.camera
        if camera is not None and camera.extrinsics is not None:
            pos = camera.extrinsics.position_m
            rot = camera.extrinsics.rotation_euler_deg
            cmd += [
                "--cam",
                f"{pos[0]},{pos[1]},{pos[2]}",
                "--rot",
                f"{rot[0]},{rot[1]},{rot[2]}",
            ]
        if camera is not None:
            intr = camera.intrinsics
            cmd += ["--f", f"{intr.fx},{intr.fy}", "--cx", str(intr.cx), "--cy", str(intr.cy)]

        self._run_backend(cmd, backend_dir, float(ctx.cfg.get("timeout_s", DEFAULT_TIMEOUT_S)))

        det_path = raw_dir / image_path.stem / "detections.json"
        box_path = raw_dir / image_path.stem / "boxes.json"
        if not det_path.exists():
            raise RuntimeError(f"pose backend wrote no detections.json at {det_path}")
        detections = json.loads(det_path.read_text())
        boxes_json = json.loads(box_path.read_text()) if box_path.exists() else None
        result = pose_boxes_from_backend(detections, boxes_json, image_ref.path)
        ctx.store.write_model(ctx.stage_name, "pose_boxes", result)

    @staticmethod
    def _run_backend(cmd: list[str], cwd: Path, timeout_s: float) -> None:
        if shutil.which("uv") is None:
            raise RuntimeError("`uv` not found on PATH; needed to run the pose backend")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"pose backend timed out after {timeout_s:.0f}s"
            ) from exc
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:])
            raise RuntimeError(
                f"pose backend failed (exit {proc.returncode}):\n{tail}"
            )
