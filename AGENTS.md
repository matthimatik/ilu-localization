# AGENTS.md

DAG orchestrator + domain logic for drone-based ILU localization (containers/trailers on drone images). Library + CLI tools, not a service.

## Commands

```bash
uv sync                          # install (dev group: pytest)
uv run pytest                    # all tests (from repo root)
uv run pytest tests/test_dag_e2e.py::test_fixture_pipeline_end_to_end
RUN_BACKEND_TESTS=1 uv run pytest tests/test_pose_backend.py tests/test_mmocr_backend.py  # needs GPU + sibling backends
uv run python tools/run_pipeline.py                                 # fixture e2e (configs/pipeline.yaml)
uv run python tools/run_pipeline.py --config configs/pipeline_pose.yaml  # real pose+mmocr backends
uv run python tools/run_pipeline.py --config configs/examples/backends_v3_drone_137.yaml  # per-scene example
uv run python tools/make_example_config.py v3_drone_137   # new example config (camera from the scene's meta.json)
uv run python tools/run_pipeline.py --no-cache
uv run python tools/visualize_run.py                 # end-result viz of newest successful run -> runs/<id>/viz/
uv run python tools/visualize_run.py --run-id <id> --open
uv run python tools/eval_run.py                    # end-to-end 3D eval vs pose.jsonl GT -> runs/<id>/eval/eval_report.json
uv run python tools/eval_run.py --run-id <id> --gt path/to/pose.jsonl --center-thres-m 2.0
uv run python tools/smoke_identify.py                # identify stage alone (needs ../ilu-identification)
```

No lint/typecheck/format tooling or CI is configured. Python `>=3.11`, managed with `uv` (`uv.lock` present). Drawing deps: `opencv-python-headless`, `plotly` (visualization only).

## Layout

Three importable packages at repo root (not `src/`): `schemas/` (pydantic models), `orchestrator/` (DAG engine + stage adapters under `orchestrator/stages/`), `core/` (pure logic: `association/`, `fusion/` incl. `iso6346`, `evaluation/` for the end-to-end metric, `visualization/` for the result renderer). CLI entrypoints live in `tools/` (scripts that `sys.path.insert` the repo root; run them as files, not as installed console scripts). Pipeline YAMLs in `configs/`. Diagram: `High Level System Diagram for ILU Localization.drawio`.

## Evaluation

- `core/evaluation/` scores a completed run against synthetic-scene ground truth (`pose.jsonl`, one line per ILU: 8 world corners + visible flags + `label` carrying the ISO 6346 code): `ground_truth.py` (parser/`parse_iso_from_label`), `geometry.py` (`box_center`, `best_corner_errors` over all 8! corner correspondences so GT/pred corner orderings need not agree — box symmetries alone do not cover the generator's reflections, `voxel_iou3d`), `matching.py` (greedy 1-to-1 by ascending center distance, `average_precision` over a center-error threshold), `metrics.py`, `api.py` (`evaluate`).
- Metrics (world 3D): precision/recall/F1 and AP at the center threshold, mean/median center error + mean corner error in metres, voxel IoU, and on matched pairs only the ISO exact / check-digit-relaxed accuracy and size accuracy. Predictions without PnP world corners are excluded from matching and listed as `pred_ids_without_world`.
- `tools/eval_run.py` is a post-run tool, **not** a pipeline stage. Writes `<run>/eval/eval_report.json` (`schemas/evaluation.py:EvalReport`); flags `--run-id`, `--runs-root`, `--gt` (dir or `pose.jsonl`; default `pose.jsonl` beside the run's image), `--center-thres-m`, `--voxel-m`, `--include-fully-occluded-gt`, `--out`. Needs a pose-backend run — fixture runs have no world corners.

## Visualization

- `core/visualization/` renders a completed run: `draw.py` (cv2 2D overlay: instance-tinted 8-corner wireframes + keypoints, OCR quads, small id badges — no metadata on the image; a detail table strip below the image keys off the badge ids; BGR, drawing style follows `../ilu-pose-estimation/keypoints/evaluate_pose.py`), `scene.py` (plotly 3D world scene + camera frustum, follows `../ilu-pose-estimation/pnp/plotting.py`; hover lists associated OCR spots, iso-code tags at top-face centres, 3D spot markers via ray–box hit when a camera exists), `report.py` (stdlib HTML report with the same record table + inlined overlay), `loader.py` (`load_run`, `find_latest_run`).
- `tools/visualize_run.py` is a post-run tool, **not** a pipeline stage (stages stay JSON-only). Writes `<run>/viz/{overlay.jpg,scene.html,report.html}`; flags `--run-id`, `--runs-root`, `--out`, `--labels {ids,all,none}`, `--spots {associated,all,none}`, `--no-scene`, `--no-report`, `--open`.
- 3D world corners/camera come from the pose backend raw output (`pose/raw/*/{detections,boxes}.json`, matched via `orchestrator.stages.pose.match_pnp`); fixture runs get centre markers only, no camera. Relative `image_path`s in artifacts resolve against the repo root.

## Pipeline architecture

Pipeline order: `load_image → (pose/mmocr/fixtures) → associate → identify → fuse`. The `identify` stage converts associated text spots into validated id codes (ISO 6346 / EN 13044-1 / registration number) via the sibling `../ilu-identification` repo; `fuse` then merges pose + identification + embeddings into `ILURecord`s. `fuse` keeps its legacy spot-picking fallback when the optional `ilu_ids` input is absent.

- Stages only exchange data as JSON artifacts via `ArtifactStore` under `runs/<run_id>/<stage>/<name>.json`. Config `inputs` use `"stage_name/artifact"` refs; a missing artifact fails the stage.
- New stage types must be registered in `orchestrator/registry.py` `_REGISTRY` (string `impl` in YAML → class). Implement the `Stage` protocol (`name` attr, `run(ctx)`); write results with `ctx.store.write_model`.
- `orchestrator/config.py:build_pipeline` injects `_base_dir` into every stage `cfg` — relative paths in YAML resolve against it (repo root when using `tools/run_pipeline.py`). Use `resolve_path(ctx, key)` for cfg path keys.
- First failed stage aborts the run (`orchestrator/dag.py`); the `manifest` is rewritten after each stage. Stage output hashing only globs `*.json` directly in the stage dir (not `raw/` subdirs).
- Content-addressed stage cache in `.cache/stages` (key: stage class + cfg JSON + input file hashes; the `pose` and `instanceseg` stages additionally hash the content of their resolved backend weights file, so a retrained/replaced checkpoint at the same path invalidates the cache). `--no-cache` bypasses; cache copies only top-level files, not subdirectories like `raw/`. The cache does **not** hash sibling source code — after changing `../ilu-identification` (or the backend repos' code), rerun with `--no-cache`.

## External backends (sibling directories, not vendored)

- Pose: `../ilu-pose-estimation` — stage shells out to `uv run pipeline/run.py` there (needs `uv` on PATH, GPU). Default weights `models/container-corners-yolov8m-1280.pt` under that repo. The PnP variant comes from stage cfg `method` (all shipped backend configs use `upright-tier`: the discrete tier-constrained fit with the free-height upright as fallback; the backend default `pnp` applies when the key is absent). The constrained fits solve from 2 visible corners up, so the shipped configs also set `min_corners: 2` (backend default is 4). Other overrides: `conf` / `kpt_conf` / `imgsz` / `min_corners` / `timeout_s`.
- Instance segmentation: `../ilu-detection` — stage shells out to `uv run python segmentation/infer.py` there (needs `uv` on PATH, GPU). Default weights `models/ilu-seg-yolov8m-1024.pt` (TRUDI v2, classes `container`/`tank_container`/`trailer`); overrides via stage cfg `weights` / `imgsz` / `conf` / `max_det` / `device`. Writes the `instances` artifact (`schemas/vision.py:Instances`); not yet consumed by `associate`/`fuse`.
- OCR: `../mmocr` — stage runs `../mmocr/.venv/bin/python tools/mmocr_runner.py` (venv must already have mmocr installed). Default det is the synthetic-v1 fine-tuned DBNet++ (`dbnetpp_resnet50-dcnv2_fpnc_30e_synth_v1`, synth val Hmean 0.699), default rec is RobustScanner fine-tuned on synthetic-v1 crops (`robustscanner_resnet31_10e_synth_v1`, synth val word_acc 0.704); both start from the TRUDI checkpoints and target **synthetic imagery only** (they perform poorly on real TRUDI frames — pass `det` / `det_weights` / `rec_weights` to use the TRUDI or stock weights instead). config/weights paths resolve relative to the mmocr repo root, override with stage cfg `det` / `det_weights` / `rec` / `rec_weights`. Runner imports mmocr inside the backend venv.
- Identification: `../ilu-identification` — stage runs `tools/identify_runner.py` (lives in **this** repo) in a **child process** via `sys.executable` (the identification lib needs only pydantic+numpy, already present). **Never import the identification package in-process**: both repos expose top-level `schemas/` and `core/` packages, which would collide. The runner does `sys.path.insert(0, backend_dir)` in its own interpreter. Override with stage cfg `python` / `runner` / `backend_dir`. Writes `schemas/identification.py:IdentifiedILUs`; `fuse` consumes it via the optional `ilu_ids` input.
- Smoke-test each adapter alone: `uv run python tools/smoke_pose.py`, `uv run python tools/smoke_instanceseg.py`, `uv run python tools/smoke_mmocr.py`, `uv run python tools/smoke_identify.py`.
- `configs/pipeline_backends.yaml` runs pose + instanceseg + mmocr on the synthetic drone scene; `configs/pipeline_pose.yaml` stays pose+mmocr only. Both include the identify stage after `associate`.
- `configs/examples/` holds one full-backend config per synthetic scene (camera from that scene's `meta.json`): `uv run python tools/make_example_config.py <scene>` writes `configs/examples/backends_<scene>.yaml` for any scene of `../ilu-pose-estimation/datasets/synthetic-data-generation` (v3_default / v3_drone / v3_human / v3_reachstacker).

## Testing

- Unit + fixture e2e always run. Backend integration tests are `@pytest.mark.skipif(RUN_BACKEND_TESTS != "1")`.
- Golden test: `tests/test_dag_e2e.py::test_fixture_pipeline_matches_golden` compares full `fuse` output to `tests/golden/ilu_records.json`. Regenerate deliberately when fusion behavior changes (run fixture pipeline with `--no-cache`, copy `fuse/ilu_records.json` over the golden file, and re-copy to `docs/data/fixture-records.json`).
- Visualization tests (`tests/test_visualization.py`) assert structure only (files written, table contents, no overlapping labels via placement helpers) — no golden images.
- Evaluation tests (`tests/test_evaluation.py`, `tests/test_eval_tool.py`) are pure + fixture: no GPU, no backend checkout; the tool test builds a canned pose-backend run in `tmp_path`.
- Builders in `tests/builders.py` (`make_spot`, `make_box`, `make_identified`, …) for vision and identification models; fixtures under `tests/fixtures/`.
- The identify stage unit tests (`tests/test_identify_stage.py`) need the sibling `../ilu-identification` checkout for the end-to-end case (skipped if absent); translation tests are pure.
