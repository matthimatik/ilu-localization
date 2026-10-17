"""Interactive 3D scene of the fused records (plotly).

Follows ../ilu-pose-estimation/pnp/plotting.py: one Scatter3d trace per box
drawn as 12 edges with None breaks, plus a camera marker and FOV frustum.

Text spotting shows up as: OCR lines in each record's hover, ISO-code tags at
the top-face centre, and 3D spot markers where the camera ray through the spot
quad hits the box (camera runs only).
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from core.association.api import quad_center
from schemas.association import Associations
from schemas.ilu import IdentificationMethod, ILURecord, ILURecords
from schemas.vision import TextSpot, TextSpots

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
# Box side faces as corner-index loops (pose-backend corner order).
FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7),
    (0, 1, 5, 4), (3, 2, 6, 7),
    (0, 3, 7, 4), (1, 2, 6, 5),
]
TOP_CORNERS = [1, 2, 5, 6]  # FLT, BLT, FRT, BRT
METHOD_STYLES = {
    IdentificationMethod.code: ("#2ca02c", "solid"),
    IdentificationMethod.embedding_rank: ("#1f77b4", "dash"),
    IdentificationMethod.unknown: ("#9e9e9e", "dash"),
}
CAM_COLOR = "#1f1f1f"
OCR_COLOR = "#ff7f0e"
TAG_COLOR = "#b30000"


def box_lines(corners: np.ndarray) -> tuple[list, list, list]:
    """x, y, z lists for one Scatter3d trace drawing the 12 edges."""
    xs: list = []
    ys: list = []
    zs: list = []
    for a, b in EDGES:
        for p in (corners[a], corners[b], (None, None, None)):
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    return xs, ys, zs


def _hover(
    record: ILURecord,
    extra: str = "",
    ocr_spots: list[TextSpot] | None = None,
) -> str:
    parts = [
        f"<b>{record.ilu_id}</b>",
        f"{record.category}, size {record.size_class or '?'}",
        f"ISO code: {record.iso_code or 'none'}",
        f"method: {record.identification_method.value}",
    ]
    confs = record.confidences or {}
    if confs:
        parts.append(
            "conf: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(confs.items()))
        )
    if record.pose is not None and record.pose.center_m is not None:
        c = record.pose.center_m
        parts.append(f"centre ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) m")
    if ocr_spots:
        parts.append(
            "OCR: "
            + "; ".join(f"{spot.text} {spot.confidence:.2f}" for spot in ocr_spots)
        )
    if extra:
        parts.append(extra)
    return "<br>".join(parts)


def add_box(
    fig: go.Figure,
    corners: np.ndarray,
    color: str,
    name: str,
    hover: str,
    dash: str = "solid",
    group: str = "",
    show_legend: bool = False,
) -> None:
    xs, ys, zs = box_lines(corners)
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"color": color, "width": 4, "dash": dash},
            name=name,
            legendgroup=group or name,
            showlegend=show_legend,
            hovertext=hover,
            hoverinfo="text",
        )
    )


def _in_face(point: np.ndarray, face: np.ndarray) -> bool:
    """Point-in-convex-quad test (consistent edge cross-product signs)."""
    n = np.cross(face[1] - face[0], face[2] - face[0])
    signs = [
        float(np.dot(np.cross(face[(i + 1) % 4] - face[i], point - face[i]), n))
        for i in range(4)
    ]
    return all(s >= -1e-6 for s in signs) or all(s <= 1e-6 for s in signs)


def spot_world(
    camera: dict, pixel: tuple[float, float], corners: np.ndarray
) -> np.ndarray | None:
    """World point where the camera ray through `pixel` hits the box, else None."""
    r_cw = np.asarray(camera["r_cw"], dtype=float)
    cam = np.asarray(camera["cam"], dtype=float)
    k = np.asarray(camera["K"], dtype=float)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    u, v = pixel
    direction = r_cw @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    best_t, best = np.inf, None
    for face in FACES:
        quad = corners[list(face)]
        n = np.cross(quad[1] - quad[0], quad[2] - quad[0])
        denom = float(n @ direction)
        if abs(denom) < 1e-9:
            continue
        t = float(n @ (quad[0] - cam)) / denom
        if t <= 1e-6 or t >= best_t:
            continue
        hit = cam + t * direction
        if _in_face(hit, quad):
            best_t, best = t, hit
    return best


def add_camera(fig: go.Figure, camera: dict, depth: float = 6.0) -> None:
    """Camera position and a small frustum showing the field of view."""
    r_cw = np.asarray(camera["r_cw"], dtype=float)
    cam = np.asarray(camera["cam"], dtype=float)
    k = np.asarray(camera["K"], dtype=float)
    width, height = camera.get("image_size") or [k[0, 2] * 2, k[1, 2] * 2]
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    corners_px = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    rays = np.c_[
        (corners_px[:, 0] - cx) / fx, (corners_px[:, 1] - cy) / fy, np.ones(4)
    ]
    far = rays * depth @ r_cw.T + cam
    xs, ys, zs = [], [], []
    for i in range(4):
        for p in (cam, far[i], far[(i + 1) % 4], (None, None, None)):
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line={"color": CAM_COLOR, "width": 2},
            name="camera",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[cam[0]], y=[cam[1]], z=[cam[2]],
            mode="markers",
            marker={"size": 5, "color": CAM_COLOR},
            showlegend=False,
            hovertext=f"camera ({cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f})",
            hoverinfo="text",
        )
    )


def plot_scene(
    records: ILURecords,
    world_by_id: dict[str, np.ndarray],
    camera: dict | None = None,
    title: str = "",
    text_spots: TextSpots | None = None,
    associations: Associations | None = None,
) -> go.Figure | None:
    """3D scene for the fused records; None when there is nothing to plot."""
    fig = go.Figure()
    first: dict[IdentificationMethod, bool] = {
        method: True for method in IdentificationMethod
    }
    spot_by_id: dict[str, list[TextSpot]] = {}
    if text_spots is not None and associations is not None:
        spot_by_id = {
            assoc.pose_box_id: [
                text_spots.spots[i]
                for i in assoc.text_spot_indices
                if 0 <= i < len(text_spots.spots)
            ]
            for assoc in associations.associations
        }

    plotted = 0
    tags: list[tuple[np.ndarray, str]] = []
    spots: list[tuple[np.ndarray, str]] = []
    for record in records.records:
        style = METHOD_STYLES.get(
            record.identification_method, ("#9e9e9e", "dash")
        )
        color, dash = style
        name = record.identification_method.value
        corners = world_by_id.get(record.ilu_id)
        ocr_spots = spot_by_id.get(record.ilu_id, [])
        if corners is not None:
            add_box(
                fig, corners, color, name, _hover(record, ocr_spots=ocr_spots),
                dash=dash, group=name, show_legend=first[record.identification_method],
            )
            if record.iso_code:
                tags.append((corners[TOP_CORNERS].mean(axis=0), record.iso_code))
            if camera is not None:
                for spot in ocr_spots:
                    world = spot_world(camera, quad_center(spot.quad), corners)
                    if world is not None:
                        spots.append((world, spot.text))
            first[record.identification_method] = False
            plotted += 1
        elif record.pose is not None and record.pose.center_m is not None:
            c = record.pose.center_m
            fig.add_trace(
                go.Scatter3d(
                    x=[c[0]], y=[c[1]], z=[c[2]],
                    mode="markers",
                    marker={"size": 5, "color": color},
                    name=name,
                    legendgroup=name,
                    showlegend=first[record.identification_method],
                    hovertext=_hover(
                        record, "centre only, no world corners", ocr_spots=ocr_spots
                    ),
                    hoverinfo="text",
                )
            )
            first[record.identification_method] = False
            plotted += 1
    if plotted == 0:
        return None
    if tags:
        fig.add_trace(
            go.Scatter3d(
                x=[p[0] for p, _ in tags],
                y=[p[1] for p, _ in tags],
                z=[p[2] for p, _ in tags],
                mode="text",
                text=[text for _, text in tags],
                textfont={"color": TAG_COLOR, "size": 11},
                name="iso codes",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if spots:
        fig.add_trace(
            go.Scatter3d(
                x=[p[0] for p, _ in spots],
                y=[p[1] for p, _ in spots],
                z=[p[2] for p, _ in spots],
                mode="markers+text",
                text=[text for _, text in spots],
                textposition="top center",
                textfont={"color": OCR_COLOR, "size": 10},
                marker={"size": 4, "color": OCR_COLOR},
                name="text spots",
                showlegend=False,
                hovertext=[f"OCR spot: {text}" for _, text in spots],
                hoverinfo="text",
            )
        )
    if camera is not None:
        add_camera(fig, camera)
    fig.update_layout(
        title=title,
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z (m)",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=750,
    )
    return fig


def write_html(fig: go.Figure, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path
