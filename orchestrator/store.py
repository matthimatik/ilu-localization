import hashlib
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


class ArtifactStore:
    def __init__(self, root: Path | str, run_id: str):
        self.root = Path(root)
        self.run_id = run_id
        self.run_dir = self.root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str) -> Path:
        d = self.run_dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path_for(self, stage: str, name: str) -> Path:
        return self.stage_dir(stage) / f"{name}.json"

    def write_model(self, stage: str, name: str, model: BaseModel) -> Path:
        return _atomic_write(self.path_for(stage, name), model)

    def write_root(self, name: str, model: BaseModel) -> Path:
        return _atomic_write(self.run_dir / f"{name}.json", model)

    def read_model(self, stage: str, name: str, model_cls: type[M]) -> M:
        return self.read_model_path(self.path_for(stage, name), model_cls)

    def read_model_path(self, path: Path | str, model_cls: type[M]) -> M:
        return model_cls.model_validate_json(Path(path).read_text())

    def exists(self, stage: str, name: str) -> bool:
        return self.path_for(stage, name).exists()

    @staticmethod
    def hash_file(path: Path | str) -> str:
        h = hashlib.sha256()
        h.update(Path(path).read_bytes())
        return h.hexdigest()


def _atomic_write(path: Path, model: BaseModel) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(model.model_dump_json(indent=2))
    os.replace(tmp, path)
    return path
