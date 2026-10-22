import time
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.cache import StageCache, stage_key
from orchestrator.stage import StageContext, StageSpec
from orchestrator.store import ArtifactStore
from schemas.run import RunManifest, StageResult, StageStatus


class Pipeline:
    def __init__(self, name: str, specs: list[StageSpec]):
        names = [s.name for s in specs]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate stage names: {names}")
        self.name = name
        self.specs = specs

    def run(
        self, store: ArtifactStore, cache: StageCache | None = None
    ) -> RunManifest:
        manifest = RunManifest(
            run_id=store.run_id,
            pipeline=self.name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        store.write_root("manifest", manifest)
        for spec in self.specs:
            result = self._run_stage(store, cache, spec)
            manifest.stages.append(result)
            store.write_root("manifest", manifest)
            if result.status == StageStatus.failed:
                manifest.status = "failed"
                store.write_root("manifest", manifest)
                return manifest
        manifest.status = "success"
        store.write_root("manifest", manifest)
        return manifest

    def _run_stage(
        self, store: ArtifactStore, cache: StageCache | None, spec: StageSpec
    ) -> StageResult:
        input_paths: dict[str, object] = {}
        for logical, ref in spec.inputs.items():
            stage_name, artifact = ref.split("/", 1)
            path = store.path_for(stage_name, artifact)
            if not path.exists():
                return StageResult(
                    stage=spec.name,
                    status=StageStatus.failed,
                    error=f"missing input artifact: {ref}",
                )
            input_paths[logical] = path

        typed_paths = {k: v for k, v in input_paths.items()}
        # Stages can declare extra files the output depends on (e.g. backend model
        # weights); they are hashed into the cache key so replacing a file in place
        # (a retrained checkpoint at the same path) invalidates the cache.
        cache_paths: dict[str, Path] = dict(typed_paths)
        declared = getattr(spec.stage, "cache_files", None)
        if callable(declared):
            for logical, path in declared(spec.cfg).items():
                if Path(path).is_file():
                    cache_paths[str(logical)] = Path(path)
        key = stage_key(spec.stage, spec.cfg, cache_paths)
        t0 = time.monotonic()
        try:
            if cache is not None and cache.get(key, spec.name, store):
                status = StageStatus.cached
            else:
                ctx = StageContext(
                    store=store,
                    stage_name=spec.name,
                    cfg=spec.cfg,
                    inputs=typed_paths,
                )
                spec.stage.run(ctx)
                if cache is not None:
                    cache.put(key, spec.name, store)
                status = StageStatus.success
        except Exception as exc:
            return StageResult(
                stage=spec.name,
                status=StageStatus.failed,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - t0,
                inputs={
                    k: ArtifactStore.hash_file(p) for k, p in typed_paths.items()
                },
            )
        outputs = {
            p.stem: ArtifactStore.hash_file(p)
            for p in sorted(store.stage_dir(spec.name).glob("*.json"))
        }
        return StageResult(
            stage=spec.name,
            status=status,
            duration_s=time.monotonic() - t0,
            inputs={k: ArtifactStore.hash_file(p) for k, p in typed_paths.items()},
            outputs=outputs,
        )
