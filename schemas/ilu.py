from enum import Enum

from pydantic import BaseModel

from schemas.image import SCHEMA_VERSION
from schemas.vision import PoseBox


class IdentificationMethod(str, Enum):
    code = "code"
    embedding_rank = "embedding_rank"
    unknown = "unknown"


class ILURecord(BaseModel):
    ilu_id: str
    category: str
    iso_code: str | None = None
    size_class: str | None = None
    identification_method: IdentificationMethod
    pose: PoseBox | None = None
    embedding_crop_id: str | None = None
    confidences: dict[str, float] = {}


class ILURecords(BaseModel):
    schema_version: str = SCHEMA_VERSION
    image_path: str
    records: list[ILURecord]
