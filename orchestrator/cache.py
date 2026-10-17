import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from orchestrator.stage import Stage
from orchestrator.store import ArtifactStore


def stage_key(stage: Stage, cfg: dict[str, Any], input_paths: dict[str, Path]) -> str:
    h = hashlib.sha256()
    h.update(type(stage).__qualname__.encode())
    h.update(stage.name.encode())
    h.update(json.dumps(cfg, sort_keys=True, default=str).encode())
    for logical in sorted(input_paths):
        h.update(logical.encode())
        h.update(ArtifactStore.hash_file(input_paths[logical]).encode())
    return h.hexdigest()[:24]


class StageCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def dir_for(self, key: str) -> Path:
        return self.root / key

    def get(self, key: str, stage: str, store: ArtifactStore) -> bool:
        src = self.dir_for(key)
        if not src.is_dir() or not any(p.is_file() for p in src.iterdir()):
            return False
        dst = store.stage_dir(stage)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        return True

    def put(self, key: str, stage: str, store: ArtifactStore) -> None:
        src = store.stage_dir(stage)
        dst = self.dir_for(key)
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
