#!/usr/bin/env python3
"""Runs ilu-identification on a batch of per-ILU spot groups.

Executed in a child process by IdentifyStage: the sibling repo's top-level
``schemas`` / ``core`` packages would collide with ilu-localization's own, so
the import must happen in an isolated interpreter (see AGENTS.md).
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Batch JSON: image_path + items[]")
    parser.add_argument("--output", required=True, help="Result JSON path")
    parser.add_argument("--backend-dir", required=True, help="Path to ../ilu-identification")
    args = parser.parse_args()

    backend_dir = Path(args.backend_dir).resolve()
    if not (backend_dir / "core" / "identify.py").exists():
        print(f"ilu-identification not found at {backend_dir}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(backend_dir))

    from core.identify import identify
    from schemas.input import ILUType, IdentificationInput, TextSpot

    batch = json.loads(Path(args.input).read_text())
    results = []
    for item in batch["items"]:
        inp = IdentificationInput(
            image_path=batch.get("image_path", ""),
            ilu_type=ILUType(item["ilu_type"]),
            spots=[TextSpot.model_validate(s) for s in item["spots"]],
        )
        results.append(
            {
                "ilu_id": item["ilu_id"],
                "ilu_type": item["ilu_type"],
                "result": identify(inp).model_dump(mode="json"),
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
