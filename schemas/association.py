from pydantic import BaseModel

from schemas.image import SCHEMA_VERSION


class Association(BaseModel):
    pose_box_id: str
    text_spot_indices: list[int]
    embedding_crop_id: str | None = None
    scores: dict[str, float] = {}


class Associations(BaseModel):
    schema_version: str = SCHEMA_VERSION
    associations: list[Association]
