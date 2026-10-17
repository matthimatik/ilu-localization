from pydantic import BaseModel, Field

from schemas.image import SCHEMA_VERSION

Point2D = tuple[float, float]


class TextSpot(BaseModel):
    quad: list[Point2D]
    text: str
    text_type: str
    confidence: float = Field(ge=0.0, le=1.0)


class TextSpots(BaseModel):
    schema_version: str = SCHEMA_VERSION
    image_path: str
    spots: list[TextSpot]


class Corner(BaseModel):
    position_px: Point2D | None
    confidence: float = Field(ge=0.0, le=1.0)


class PoseBox(BaseModel):
    box_id: str
    corners_px: list[Corner]
    size_class: str
    center_m: tuple[float, float, float] | None = None
    method: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class PoseBoxes(BaseModel):
    schema_version: str = SCHEMA_VERSION
    image_path: str
    boxes: list[PoseBox]


class Embedding(BaseModel):
    crop_id: str
    vector: list[float]
    source: str = "unknown"


class Embeddings(BaseModel):
    schema_version: str = SCHEMA_VERSION
    embeddings: list[Embedding]
