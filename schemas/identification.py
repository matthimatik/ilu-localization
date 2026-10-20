from pydantic import BaseModel

from schemas.image import SCHEMA_VERSION


class IdSizeType(BaseModel):
    """Slim mirror of ilu-identification's decoded size/type."""

    code: str
    length_m: float | None = None
    height_m: float | None = None
    body_kind: str = "other"
    high_cube: bool = False


class IdentifiedILU(BaseModel):
    ilu_id: str
    ilu_type: str
    id_code: str | None = None
    id_scheme: str = "none"
    id_valid: bool = False
    id_confidence: float = 0.0
    method: str = "unknown"
    size_type: IdSizeType | None = None
    warnings: list[str] = []


class IdentifiedILUs(BaseModel):
    schema_version: str = SCHEMA_VERSION
    image_path: str = ""
    results: list[IdentifiedILU]
