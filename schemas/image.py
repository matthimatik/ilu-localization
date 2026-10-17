from pydantic import BaseModel

SCHEMA_VERSION = "1"


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class CameraExtrinsics(BaseModel):
    position_m: tuple[float, float, float]
    rotation_euler_deg: tuple[float, float, float]


class CameraModel(BaseModel):
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics | None = None


class ImageRef(BaseModel):
    schema_version: str = SCHEMA_VERSION
    path: str
    camera: CameraModel | None = None
