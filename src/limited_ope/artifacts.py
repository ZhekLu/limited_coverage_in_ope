"""Atomic result storage, source provenance, resumability and artifact integrity."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from limited_ope.config import ExperimentConfig


def timestamp() -> str:
    """Return a timezone-explicit UTC timestamp."""
    return datetime.now(UTC).isoformat()


def file_digest(path: Path) -> str:
    """Compute SHA-256 for a saved file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value: dict[str, Any]) -> str:
    """Hash a canonical JSON object independent of indentation or key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a JSON object, rejecting nonfinite numeric output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class ArtifactStore:
    """Save one run without overwriting another experiment or mixing source versions.

    :param config: Frozen configuration snapshot.
    :param project_root: Source repository containing the lockfile.
    """

    def __init__(self, config: ExperimentConfig, project_root: Path) -> None:
        self.config, self.project_root = config, project_root
        self.root = config.run_dir
        paths = sorted((project_root / "src" / "limited_ope").glob("*.py"))
        paths += [project_root / name for name in ("pyproject.toml", "uv.lock", "docs/protocol.md")]
        self.source_files = {str(p.relative_to(project_root)): file_digest(p) for p in paths}
        self.source_digest = json_digest(self.source_files)
        self.config_digest = json_digest(config.to_dict())

    def prepare(self, resume: bool) -> dict[str, Any]:
        """Initialize provenance or verify that resumption matches the original study."""
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"Run exists: {self.root}; use --resume or a new config name")
            manifest: dict[str, Any] = json.loads(manifest_path.read_text())
            if manifest["config_sha256"] != self.config_digest:
                raise ValueError("Cannot resume: configuration differs from saved run")
            if manifest["source_sha256"] != self.source_digest:
                raise ValueError("Cannot resume: source/protocol/lockfile differs from saved run")
            return manifest
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("Nonempty run directory without a manifest; use another name")
        self.root.mkdir(parents=True, exist_ok=True)
        snapshot = self.root / "provenance" / "source"
        for relative in self.source_files:
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.project_root / relative, destination)
        git: dict[str, str] = {}
        for label, args in (
            ("commit", ["rev-parse", "HEAD"]),
            ("status", ["status", "--porcelain"]),
            (
                "diff",
                ["diff", "HEAD", "--", "src", "configs", "docs", "pyproject.toml", "Makefile"],
            ),
        ):
            result = subprocess.run(
                ["git", *args],
                cwd=self.project_root,
                text=True,
                capture_output=True,
                check=False,
            )
            git[label] = result.stdout.strip() if result.returncode == 0 else "unavailable"
        dependencies = {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        }
        manifest = {
            "schema_version": 1,
            "created_utc": timestamp(),
            "status": "running",
            "config_sha256": self.config_digest,
            "source_sha256": self.source_digest,
            "source_files": self.source_files,
            "git": git,
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependencies,
            "seed_scheme": "SHA256 labels + SeedSequence -> 128-bit PCG64 seeds; version 1",
            "estimand": "finite-horizon expected discounted return; gamma=1 is success by H",
        }
        atomic_json(manifest_path, manifest)
        atomic_json(self.root / "config.json", self.config.to_dict())
        return manifest

    def record_path(self, scope: str, cell_id: str) -> Path:
        """Return a stable per-dataset checkpoint path."""
        return self.root / "records" / scope / f"{cell_id}.json"

    def completed(self, scope: str, cell_id: str) -> bool:
        """Validate every dependent file before skipping a completed cell."""
        path = self.record_path(scope, cell_id)
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        for relative, digest in record["artifact_sha256"].items():
            artifact = self.root / relative
            if not artifact.exists() or file_digest(artifact) != digest:
                raise ValueError(f"Checkpoint artifact missing or corrupted: {artifact}")
        return True

    def write_record(self, scope: str, cell_id: str, record: dict[str, Any]) -> None:
        """Commit a cell only after its referenced artifacts have been saved."""
        atomic_json(self.record_path(scope, cell_id), record)

    def log(self, message: str) -> None:
        """Append a timestamped event to the run's readable audit trail."""
        with (self.root / "run.log").open("a") as handle:
            handle.write(f"{timestamp()} {message}\n")

    def update_manifest(self, **updates: str | int | float) -> None:
        """Atomically update run status after a significant lifecycle event."""
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update(updates)
        atomic_json(path, manifest)
