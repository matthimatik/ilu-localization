# docs/ — GitHub Pages showcase

Static single-page showcase for this repo, served from the `master` branch
(GitHub → Settings → Pages → Deploy from branch → `master` /`/docs`).

No build step: `index.html` + plain CSS; diagrams render via mermaid (CDN) and
the 3D scenes are self-contained plotly pages loading plotly from its CDN.

## Assets are snapshots

`runs/` is gitignored, so everything under `assets/img/` and `demos/` was
copied/downscaled from local run outputs and **will not refresh itself**.

| File | Source |
|---|---|
| `assets/img/backends-overlay.jpg` | `runs/20261018-081350-511036/viz/overlay.jpg` |
| `assets/img/backends-input.jpg` | `../ilu-pose-estimation/datasets/synthetic-data-generation/v3_drone_0/rgb.jpg` |
| `assets/img/real-photo-overlay.jpg` | `runs/20261018-002257-358607/viz/overlay.jpg` |
| `assets/img/real-photo-input.jpg` | `/data/8hirsch/trudi/data/20250102_121248.jpg` (local) |
| `assets/img/fixture-overlay.jpg` | `runs/20261017-161314-350892/viz/overlay.jpg` |
| `assets/img/fixture-input.png` | `tests/fixtures/image.png` (committed) |
| `demos/backends-scene.html` | `runs/20261018-081350-511036/viz/scene.html` |
| `demos/fixture-scene.html` | `runs/20261017-161314-350892/viz/scene.html` |
| `data/fixture-records.json` | `tests/golden/ilu_records.json` (committed) |

## Regenerating

```bash
# 1. re-run the pipelines and render visualizations
uv run python tools/run_pipeline.py                                # fixture run
uv run python tools/run_pipeline.py --config configs/pipeline_backends.yaml
uv run python tools/visualize_run.py --run-id <run_id>

# 2. downscale overlays to web width (1600px, q=82) with the project venv
.venv/bin/python - <<'EOF'
import cv2, os
def save(src, dst, width=1600, q=82):
    img = cv2.imread(src); h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, round(h*width/w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    print(dst, os.path.getsize(dst)//1024, "KB")
save("runs/<id>/viz/overlay.jpg", "docs/assets/img/<name>.jpg")
EOF

# 3. copy the self-contained scene pages
cp runs/<id>/viz/scene.html docs/demos/<name>-scene.html
```

Notes:

- The real-photo run (`20261018-002257`) has **no 3D scene**: no camera
  calibration / world corners in that run, so `visualize_run.py` skips it and
  the showcase shows the overlay only.
- Stats and result tables in `index.html` are hand-transcribed from each run's
  `fuse/ilu_records.json` — update them if you re-run with new weights/config.
- The high-level diagram in `index.html` is a mermaid redraw of
  `High Level System Diagram for ILU Localization.drawio`.

## Preview locally

```bash
python3 -m http.server -d docs 8000   # → http://localhost:8000
```
