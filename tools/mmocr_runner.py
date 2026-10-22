#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MMOCR det+rec runner (runs inside the mmocr backend venv)"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--det",
        default="configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_30e_synth_v1.py",
    )
    parser.add_argument(
        "--det-weights",
        default=(
            "work_dirs/dbnetpp_synth_v1_30e/"
            "dbnetpp_resnet50-dcnv2_fpnc_30e_synth_v1.pth"
        ),
    )
    parser.add_argument("--rec", default="RobustScanner")
    parser.add_argument(
        "--rec-weights",
        default=(
            "work_dirs/robustscanner_synth_v1_10e/"
            "robustscanner_resnet31_10e_synth_v1.pth"
        ),
        help="recognition checkpoint path (release-layout .pth); "
        "set to an empty string to use the stock RobustScanner weights",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from mmocr.apis.inferencers import MMOCRInferencer
    from mmocr.utils import bbox2poly, poly2bbox

    inferencer = MMOCRInferencer(
        det=args.det,
        det_weights=args.det_weights or None,
        rec=args.rec,
        rec_weights=args.rec_weights or None,
        device=args.device,
    )
    result = inferencer(args.image, batch_size=1)
    pred = result["predictions"][0]
    polygons = pred.get("det_polygons") or []
    det_scores = pred.get("det_scores") or []
    rec_texts = pred.get("rec_texts") or []
    rec_scores = pred.get("rec_scores") or []

    spots = []
    for i, poly in enumerate(polygons):
        quad_flat = bbox2poly(poly2bbox(poly))
        quad = [
            [float(quad_flat[2 * j]), float(quad_flat[2 * j + 1])]
            for j in range(4)
        ]
        det_score = float(det_scores[i]) if i < len(det_scores) else 1.0
        rec_score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
        text = rec_texts[i] if i < len(rec_texts) else ""
        spots.append(
            {
                "quad": quad,
                "text": text,
                "det_score": det_score,
                "rec_score": rec_score,
                "score": det_score * rec_score,
            }
        )
    Path(args.out).write_text(
        json.dumps({"image": args.image, "spots": spots}, indent=1)
    )
    print(f"{len(spots)} text spots -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
