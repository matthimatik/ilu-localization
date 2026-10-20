import json
import subprocess
import sys
from pathlib import Path

from orchestrator.stage import StageContext
from orchestrator.stages.util import load_model, resolve_path
from schemas.association import Associations
from schemas.identification import IdentifiedILU, IdentifiedILUs, IdSizeType
from schemas.vision import PoseBox, PoseBoxes, TextSpot, TextSpots

DEFAULT_TIMEOUT_S = 120.0


def ilu_type_for(box: PoseBox, spots: list[TextSpot]) -> str:
    """trailer | tank_container | container — same rule as core.fusion category."""
    if box.size_class == "trailer":
        return "trailer"
    for spot in spots:
        if "tank" in f"{spot.text} {spot.text_type}".lower():
            return "tank_container"
    return "container"


def items_for(
    associations: Associations,
    text_spots: TextSpots,
    pose_boxes: PoseBoxes,
) -> tuple[list[dict], list[IdentifiedILU]]:
    """Per-association subprocess payloads plus local placeholders for empty groups."""
    boxes_by_id = {box.box_id: box for box in pose_boxes.boxes}
    items: list[dict] = []
    placeholders: list[IdentifiedILU] = []
    for assoc in associations.associations:
        box = boxes_by_id[assoc.pose_box_id]
        spots = [
            text_spots.spots[i]
            for i in assoc.text_spot_indices
            if 0 <= i < len(text_spots.spots)
        ]
        ilu_type = ilu_type_for(box, spots)
        if not spots:
            placeholders.append(
                IdentifiedILU(ilu_id=box.box_id, ilu_type=ilu_type, method="unknown")
            )
            continue
        items.append(
            {
                "ilu_id": box.box_id,
                "ilu_type": ilu_type,
                "spots": [
                    {
                        "quad": [[p[0], p[1]] for p in spot.quad],
                        "text": spot.text,
                        "score": spot.confidence,
                    }
                    for spot in spots
                ],
            }
        )
    return items, placeholders


def results_from_runner(raw: dict, image_path: str) -> IdentifiedILUs:
    """Translate the runner's batch output into an IdentifiedILUs artifact."""
    results: list[IdentifiedILU] = []
    for entry in raw["results"]:
        r = entry["result"]
        size_type = None
        if r.get("size_type"):
            st = r["size_type"]
            size_type = IdSizeType(
                code=st["code"],
                length_m=st.get("length_m"),
                height_m=st.get("height_m"),
                body_kind=st.get("body_kind", "other"),
                high_cube=st.get("high_cube", False),
            )
        results.append(
            IdentifiedILU(
                ilu_id=entry["ilu_id"],
                ilu_type=entry.get("ilu_type", "container"),
                id_code=r.get("id_code"),
                id_scheme=r.get("id_scheme", "none"),
                id_valid=r.get("id_valid", False),
                id_confidence=r.get("id_confidence", 0.0),
                method=r.get("method", "unknown"),
                size_type=size_type,
                warnings=r.get("warnings", []),
            )
        )
    return IdentifiedILUs(image_path=image_path, results=results)


class IdentifyStage:
    name = "identify"

    def run(self, ctx: StageContext) -> None:
        associations = load_model(ctx, "associations", Associations)
        text_spots = load_model(ctx, "text_spots", TextSpots)
        pose_boxes = load_model(ctx, "pose_boxes", PoseBoxes)

        backend_dir = resolve_path(ctx, "backend_dir")
        if not backend_dir.is_dir():
            raise FileNotFoundError(f"ilu-identification backend not found: {backend_dir}")
        runner = (
            resolve_path(ctx, "runner")
            if "runner" in ctx.cfg
            else Path(ctx.cfg["_base_dir"]) / "tools" / "identify_runner.py"
        )
        python = ctx.cfg.get("python", sys.executable)

        items, placeholders = items_for(associations, text_spots, pose_boxes)

        results: list[IdentifiedILU] = list(placeholders)
        if items:
            raw_dir = ctx.store.stage_dir(ctx.stage_name) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            in_path = raw_dir / "identify_input.json"
            out_path = raw_dir / "identify_output.json"
            in_path.write_text(
                json.dumps({"image_path": text_spots.image_path, "items": items}, indent=2)
            )
            cmd = [
                str(python),
                str(runner),
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--backend-dir",
                str(backend_dir),
            ]
            self._run_backend(
                cmd, backend_dir, float(ctx.cfg.get("timeout_s", DEFAULT_TIMEOUT_S))
            )
            if not out_path.exists():
                raise RuntimeError(f"identify runner wrote no output at {out_path}")
            raw = json.loads(out_path.read_text())
            runner_results = results_from_runner(raw, text_spots.image_path)
            results.extend(runner_results.results)

        # Preserve association order.
        order = {assoc.pose_box_id: i for i, assoc in enumerate(associations.associations)}
        results.sort(key=lambda r: order.get(r.ilu_id, len(order)))
        artifact = IdentifiedILUs(image_path=text_spots.image_path, results=results)
        ctx.store.write_model(ctx.stage_name, "ilu_ids", artifact)

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
                f"identify backend timed out after {timeout_s:.0f}s"
            ) from exc
        if proc.returncode != 0:
            tail = "\n".join(
                (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:]
            )
            raise RuntimeError(
                f"identify backend failed (exit {proc.returncode}):\n{tail}"
            )
