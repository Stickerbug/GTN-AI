from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


RUNTIME_MODULES = (
    "__init__.py",
    "belief_sampling.py",
    "client.py",
    "contextual_value.py",
    "deck_prior.py",
    "diagnostics.py",
    "environment.py",
    "features.py",
    "game_imports.py",
    "historical_aggregate.py",
    "live_worker.py",
    "neural_model.py",
    "observation.py",
    "policies.py",
    "protocol.py",
    "random_scope.py",
    "rollout_search.py",
    "structured_features.py",
    "structured_model.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _add_file(archive: tarfile.TarFile, source: Path, target: str) -> None:
    info = archive.gettarinfo(str(source), arcname=target)
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def _add_bytes(archive: tarfile.TarFile, content: bytes, target: str) -> None:
    info = tarfile.TarInfo(target)
    info.size = len(content)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    archive.addfile(info, io.BytesIO(content))


def build_bundle(root: Path, checkpoint: Path, output: Path) -> dict[str, object]:
    package_root = root / "gtn_ai"
    missing = [name for name in RUNTIME_MODULES if not (package_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing runtime modules: {', '.join(missing)}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    manifest: dict[str, object] = {
        "format": 1,
        "source_commit": _source_commit(root),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "runtime_modules": list(RUNTIME_MODULES),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        for name in RUNTIME_MODULES:
            _add_file(archive, package_root / name, f"gtn-ai-runtime/gtn_ai/{name}")
        _add_file(
            archive,
            checkpoint,
            f"gtn-ai-runtime/models/{checkpoint.name}",
        )
        _add_bytes(
            archive,
            (json.dumps(manifest, ensure_ascii=True, indent=2) + "\n").encode("ascii"),
            "gtn-ai-runtime/runtime-manifest.json",
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the production-only GTN-AI inference bundle."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/structured-v2-search-dagger-v2.epoch-06.pt",
    )
    parser.add_argument("--output", default="dist/gtn-ai-runtime.tar.gz")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    manifest = build_bundle(root, checkpoint.resolve(), output.resolve())
    print(json.dumps({"output": str(output), **manifest}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
