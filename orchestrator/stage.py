from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from orchestrator.store import ArtifactStore


@dataclass
class StageContext:
    store: ArtifactStore
    stage_name: str
    cfg: dict[str, Any]
    inputs: dict[str, Path]


class Stage(Protocol):
    name: str

    def run(self, ctx: StageContext) -> None: ...


@dataclass
class StageSpec:
    name: str
    stage: Stage
    inputs: dict[str, str] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)
