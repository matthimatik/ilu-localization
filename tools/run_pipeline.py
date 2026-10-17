#!/usr/bin/env python3
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from orchestrator.cache import StageCache
from orchestrator.config import build_pipeline
from orchestrator.store import ArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ILU localization pipeline")
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "pipeline.yaml")
    )
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    pipeline = build_pipeline(args.config, base_dir=REPO_ROOT)
    store = ArtifactStore(args.runs_root, run_id)
    cache = None if args.no_cache else StageCache(REPO_ROOT / ".cache" / "stages")

    manifest = pipeline.run(store, cache)
    print(f"run_id: {run_id}")
    print(f"status: {manifest.status}")
    for result in manifest.stages:
        line = f"  {result.stage}: {result.status.value} ({result.duration_s:.2f}s)"
        if result.error:
            line += f" — {result.error}"
        print(line)
    if manifest.status == "success":
        records_path = store.path_for("fuse", "ilu_records")
        print(f"records: {records_path}")
    return 0 if manifest.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
