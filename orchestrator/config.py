from pathlib import Path

import yaml

from orchestrator.dag import Pipeline
from orchestrator.registry import create_stage
from orchestrator.stage import StageSpec


def build_pipeline(config_path: Path | str, base_dir: Path | str) -> Pipeline:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())
    base_dir_str = str(base_dir)
    specs: list[StageSpec] = []
    for entry in raw["stages"]:
        cfg = dict(entry.get("cfg") or {})
        cfg["_base_dir"] = base_dir_str
        specs.append(
            StageSpec(
                name=entry["name"],
                stage=create_stage(entry["impl"]),
                inputs=entry.get("inputs") or {},
                cfg=cfg,
            )
        )
    return Pipeline(name=raw.get("pipeline", "pipeline"), specs=specs)
