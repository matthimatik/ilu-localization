import json
import shutil
import subprocess
from pathlib import Path

from orchestrator.stage import StageContext
from orchestrator.stages.util import load_model, resolve_path
from schemas.image import ImageRef
from schemas.vision import Instance, Instances

DEFAULT_WEIGHTS = "models/ilu-seg-yolov8m-1024.pt"
DEFAULT_TIMEOUT_S = 300.0


def instances_from_runner(data: dict, image_path: str) -> Instances:
    built: list[Instance] = []
    for index, entry in enumerate(data["instances"]):
        built.append(
            Instance(
                instance_id=f"seg-{index}",
                label=str(entry["cls"]),
                confidence=min(1.0, max(0.0, float(entry["conf"]))),
                bbox_px=tuple(float(v) for v in entry["bbox_xyxy"]),
                polygon_px=[(float(p[0]), float(p[1])) for p in entry["polygon"]],
            )
        )
    return Instances(image_path=image_path, instances=built)


class InstancesegBackendStage:
    name = "instanceseg"

    def run(self, ctx: StageContext) -> None:
        image_ref = load_model(ctx, "image", ImageRef)
        backend_dir = resolve_path(ctx, "backend_dir")
        if not backend_dir.is_dir():
            raise FileNotFoundError(f"instance-segmentation backend not found: {backend_dir}")

        image_path = Path(image_ref.path)
        if not image_path.is_absolute():
            image_path = Path(ctx.cfg["_base_dir"]) / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        weights = Path(ctx.cfg.get("weights", DEFAULT_WEIGHTS))
        if not weights.is_absolute():
            weights = backend_dir / weights

        out_path = ctx.store.stage_dir(ctx.stage_name) / "raw" / "instances.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "uv",
            "run",
            "python",
            "segmentation/infer.py",
            str(image_path),
            "--out",
            str(out_path),
            "--weights",
            str(weights),
        ]
        for flag, key in (
            ("--imgsz", "imgsz"),
            ("--conf", "conf"),
            ("--device", "device"),
            ("--max-det", "max_det"),
        ):
            if key in ctx.cfg:
                cmd += [flag, str(ctx.cfg[key])]

        self._run_backend(cmd, backend_dir, float(ctx.cfg.get("timeout_s", DEFAULT_TIMEOUT_S)))
        if not out_path.exists():
            raise RuntimeError(f"instance-segmentation runner wrote no output at {out_path}")
        data = json.loads(out_path.read_text())
        result = instances_from_runner(data, image_ref.path)
        ctx.store.write_model(ctx.stage_name, "instances", result)

    @staticmethod
    def _run_backend(cmd: list[str], cwd: Path, timeout_s: float) -> None:
        if shutil.which("uv") is None:
            raise RuntimeError(
                "`uv` not found on PATH; needed to run the instance-segmentation backend"
            )
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
                f"instance-segmentation backend timed out after {timeout_s:.0f}s"
            ) from exc
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:])
            raise RuntimeError(
                f"instance-segmentation backend failed (exit {proc.returncode}):\n{tail}"
            )
