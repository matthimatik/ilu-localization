from pydantic import BaseModel

from schemas.image import SCHEMA_VERSION


class GTBox(BaseModel):
    """One ground-truth ILU from a synthetic scene's pose.jsonl."""

    index: int
    world_corners: list[tuple[float, float, float]]
    visible: list[bool]
    size_class: str | None = None
    label: str = ""
    iso_code: str | None = None


class GTBoxes(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: str = ""
    boxes: list[GTBox]


class MatchedPair(BaseModel):
    gt_index: int
    pred_ilu_id: str
    center_error_m: float
    corner_errors_m: list[float]
    mean_corner_error_m: float
    max_corner_error_m: float
    iou3d: float | None = None
    pose_confidence: float | None = None
    gt_iso_code: str | None = None
    pred_iso_code: str | None = None
    id_correct: bool | None = None
    id_relaxed_correct: bool | None = None
    gt_size_class: str | None = None
    pred_size_class: str | None = None
    size_correct: bool | None = None


class SceneMetrics(BaseModel):
    n_gt: int
    n_gt_visible: int
    n_pred: int
    n_pred_with_world: int
    n_pred_without_world: int
    n_matched: int
    center_threshold_m: float
    precision: float
    recall: float
    f1: float
    ap_center: float | None = None
    mean_center_error_m: float | None = None
    median_center_error_m: float | None = None
    mean_corner_error_m: float | None = None
    id_accuracy: float | None = None
    id_relaxed_accuracy: float | None = None
    size_accuracy: float | None = None


class EvalReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    image_path: str = ""
    gt_source: str = ""
    params: dict[str, float] = {}
    pairs: list[MatchedPair] = []
    metrics: SceneMetrics
    unmatched_gt_indices: list[int] = []
    unmatched_pred_ids: list[str] = []
    pred_ids_without_world: list[str] = []
