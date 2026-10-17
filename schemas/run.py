from enum import Enum

from pydantic import BaseModel


class StageStatus(str, Enum):
    success = "success"
    cached = "cached"
    failed = "failed"


class StageResult(BaseModel):
    stage: str
    status: StageStatus
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    error: str | None = None
    duration_s: float = 0.0


class RunManifest(BaseModel):
    run_id: str
    pipeline: str
    created_at: str
    config_hash: str = ""
    status: str = "running"
    stages: list[StageResult] = []
