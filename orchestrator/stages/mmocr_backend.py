import json
import shutil
import subprocess
from pathlib import Path

from core.fusion.iso6346 import ISO_6346_ID_REGEX, normalize
from orchestrator.stage import StageContext
from orchestrator.stages.util import load_model, resolve_path
from schemas.image import ImageRef
from schemas.vision import TextSpot, TextSpots

DEFAULT_DET = "configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_30e_synth_v1.py"
DEFAULT_DET_WEIGHTS = (
    "work_dirs/dbnetpp_synth_v1_30e/dbnetpp_resnet50-dcnv2_fpnc_30e_synth_v1.pth"
)
DEFAULT_REC = "RobustScanner"
DEFAULT_REC_WEIGHTS = (
    "work_dirs/robustscanner_synth_v1_10e/robustscanner_resnet31_10e_synth_v1.pth"
)
DEFAULT_TIMEOUT_S = 600.0


def resolve_backend_model(value: str, backend_dir: Path) -> str:
    """Resolve a config/weights path against ``backend_dir``; pass aliases through."""
    path = Path(value)
    if path.suffix in {".py", ".pth", ".pt", ".ckpt"} and not path.is_absolute():
        return str(backend_dir / path)
    return value


def text_spots_from_runner(data: dict, image_path: str) -> TextSpots:
    spots: list[TextSpot] = []
    for entry in data["spots"]:
        text = str(entry["text"]).strip()
        norm = normalize(text)
        text_type = "id_iso6346" if ISO_6346_ID_REGEX.match(norm) else "unknown"
        spots.append(
            TextSpot(
                quad=[(float(p[0]), float(p[1])) for p in entry["quad"]],
                text=text,
                text_type=text_type,
                confidence=min(1.0, max(0.0, float(entry["score"]))),
            )
        )
    return TextSpots(image_path=image_path, spots=spots)


class MmocrBackendStage:
    name = "mmocr"

    def run(self, ctx: StageContext) -> None:
        image_ref = load_model(ctx, "image", ImageRef)
        backend_dir = resolve_path(ctx, "backend_dir")
        python = backend_dir / ".venv" / "bin" / "python"
        if not python.exists():
            raise RuntimeError(
                f"mmocr venv not found at {python}; set it up per the mmocr repo docs"
            )
        runner = resolve_path(ctx, "runner") if "runner" in ctx.cfg else (
            Path(ctx.cfg["_base_dir"]) / "tools" / "mmocr_runner.py"
        )

        image_path = Path(image_ref.path)
        if not image_path.is_absolute():
            image_path = Path(ctx.cfg["_base_dir"]) / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        out_path = ctx.store.stage_dir(ctx.stage_name) / "raw" / "ocr.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        det = resolve_backend_model(ctx.cfg.get("det", DEFAULT_DET), backend_dir)
        det_weights = ctx.cfg.get("det_weights", DEFAULT_DET_WEIGHTS)
        cmd = [
            str(python),
            str(runner),
            "--image",
            str(image_path),
            "--out",
            str(out_path),
            "--det",
            det,
            "--rec",
            str(ctx.cfg.get("rec", DEFAULT_REC)),
        ]
        if det_weights:
            cmd += [
                "--det-weights",
                resolve_backend_model(str(det_weights), backend_dir),
            ]
        rec_weights = ctx.cfg.get("rec_weights", DEFAULT_REC_WEIGHTS)
        if rec_weights:
            cmd += [
                "--rec-weights",
                resolve_backend_model(str(rec_weights), backend_dir),
            ]
        if "device" in ctx.cfg:
            cmd += ["--device", str(ctx.cfg["device"])]

        self._run_backend(
            cmd,
            backend_dir,
            float(ctx.cfg.get("timeout_s", DEFAULT_TIMEOUT_S)),
        )
        if not out_path.exists():
            raise RuntimeError(f"mmocr runner wrote no output at {out_path}")
        data = json.loads(out_path.read_text())
        result = text_spots_from_runner(data, image_ref.path)
        ctx.store.write_model(ctx.stage_name, "text_spots", result)

    @staticmethod
    def _run_backend(cmd: list[str], cwd: Path, timeout_s: float) -> None:
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
                f"mmocr backend timed out after {timeout_s:.0f}s"
            ) from exc
        if proc.returncode != 0:
            tail = "\n".join(
                (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:]
            )
            raise RuntimeError(
                f"mmocr backend failed (exit {proc.returncode}):\n{tail}"
            )
