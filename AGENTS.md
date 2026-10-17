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
uv run python tools/run_pipeline.py --no-cache
uv run python tools/visualize_run.py                 # end-result viz of newest successful run -> runs/<id>/viz/
uv run python tools/visualize_run.py --run-id <id> --open
```

No lint/typecheck/format tooling or CI is configured. Python `>=3.11`, managed with `uv` (`uv.lock` present). Drawing deps: `opencv-python-headless`, `plotly` (visualization only).

## Layout

Three importable packages at repo root (not `src/`): `schemas/` (pydantic models), `orchestrator/` (DAG engine + stage adapters under `orchestrator/stages/`), `core/` (pure logic: `association/`, `fusion/` incl. `iso6346`, `visualization/` for the result renderer). CLI entrypoints live in `tools/` (scripts that `sys.path.insert` the repo root; run them as files, not as installed console scripts). Pipeline YAMLs in `configs/`. Diagram: `High Level System Diagram for ILU Localization.drawio`.

## Visualization

- `core/visualization/` renders a completed run: `draw.py` (cv2 2D overlay: instance-tinted 8-corner wireframes + keypoints, OCR quads, small id badges — no metadata on the image; a detail table strip below the image keys off the badge ids; BGR, drawing style follows `../ilu-pose-estimation/keypoints/evaluate_pose.py`), `scene.py` (plotly 3D world scene + camera frustum, follows `../ilu-pose-estimation/pnp/plotting.py`; hover lists associated OCR spots, iso-code tags at top-face centres, 3D spot markers via ray–box hit when a camera exists), `report.py` (stdlib HTML report with the same record table + inlined overlay), `loader.py` (`load_run`, `find_latest_run`).
- `tools/visualize_run.py` is a post-run tool, **not** a pipeline stage (stages stay JSON-only). Writes `<run>/viz/{overlay.jpg,scene.html,report.html}`; flags `--run-id`, `--runs-root`, `--out`, `--labels {ids,all,none}`, `--spots {associated,all,none}`, `--no-scene`, `--no-report`, `--open`.
- 3D world corners/camera come from the pose backend raw output (`pose/raw/*/{detections,boxes}.json`, matched via `orchestrator.stages.pose.match_pnp`); fixture runs get centre markers only, no camera. Relative `image_path`s in artifacts resolve against the repo root.

## Pipeline architecture

- Stages only exchange data as JSON artifacts via `ArtifactStore` under `runs/<run_id>/<stage>/<name>.json`. Config `inputs` use `"stage_name/artifact"` refs; a missing artifact fails the stage.
- New stage types must be registered in `orchestrator/registry.py` `_REGISTRY` (string `impl` in YAML → class). Implement the `Stage` protocol (`name` attr, `run(ctx)`); write results with `ctx.store.write_model`.
- `orchestrator/config.py:build_pipeline` injects `_base_dir` into every stage `cfg` — relative paths in YAML resolve against it (repo root when using `tools/run_pipeline.py`). Use `resolve_path(ctx, key)` for cfg path keys.
- First failed stage aborts the run (`orchestrator/dag.py`); the `manifest` is rewritten after each stage. Stage output hashing only globs `*.json` directly in the stage dir (not `raw/` subdirs).
- Content-addressed stage cache in `.cache/stages` (key: stage class + cfg JSON + input file hashes). `--no-cache` bypasses; cache copies only top-level files, not subdirectories like `raw/`.

## External backends (sibling directories, not vendored)

- Pose: `../ilu-pose-estimation` — stage shells out to `uv run pipeline/run.py` there (needs `uv` on PATH, GPU). Default weights `models/container-corners-yolov8m-1280.pt` under that repo.
- OCR: `../mmocr` — stage runs `../mmocr/.venv/bin/python tools/mmocr_runner.py` (venv must already have mmocr installed; default det/rec: DBNet++ + RobustScanner). Runner imports mmocr inside the backend venv.
- Smoke-test each adapter alone: `uv run python tools/smoke_pose.py`, `uv run python tools/smoke_mmocr.py`.

## Testing

- Unit + fixture e2e always run. Backend integration tests are `@pytest.mark.skipif(RUN_BACKEND_TESTS != "1")`.
- Golden test: `tests/test_dag_e2e.py::test_fixture_pipeline_matches_golden` compares full `fuse` output to `tests/golden/ilu_records.json`. Regenerate deliberately when fusion behavior changes (run fixture pipeline, copy `fuse/ilu_records.json` over the golden file).
- Visualization tests (`tests/test_visualization.py`) assert structure only (files written, table contents, no overlapping labels via placement helpers) — no golden images.
- Builders in `tests/builders.py` (`make_spot`, `make_box`, …) for vision models; fixtures under `tests/fixtures/`.
