from __future__ import annotations

import gzip
import json
import os
import pickle
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .protocol import Action


DIAGNOSTIC_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _managed_diagnostic_entries(root: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    if not root.is_dir():
        return entries
    for child in root.iterdir():
        if child.is_dir() and child.name != "exports" and (child / "manifest.json").is_file():
            entries.append((child, child.name))
    exports = root / "exports"
    if exports.is_dir():
        for child in exports.glob("*.gtnai.zip"):
            entries.append((child, child.name.removesuffix(".gtnai.zip")))
    return entries


def _entry_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def _entry_timestamp(path: Path) -> float:
    manifest = path / "manifest.json" if path.is_dir() else None
    if manifest is not None and manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            value = payload.get("finished_at") or payload.get("created_at")
            if value:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return path.stat().st_mtime


def cleanup_diagnostic_storage(
    root: str | Path,
    *,
    retention_days: float = 14.0,
    max_bytes: int = 2 * 1024 * 1024 * 1024,
    protected_session_ids: Iterable[str] = (),
    now: float | None = None,
) -> dict[str, int]:
    """Bound finished diagnostic storage without touching active sessions."""

    storage_root = Path(root).resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    protected = {str(value) for value in protected_session_ids}
    current_time = time.time() if now is None else float(now)
    cutoff = (
        current_time - max(0.0, float(retention_days)) * 86400.0
        if float(retention_days) > 0
        else None
    )
    removed_entries = 0
    removed_bytes = 0

    def remove_entry(path: Path) -> None:
        nonlocal removed_entries, removed_bytes
        size = _entry_size(path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        removed_entries += 1
        removed_bytes += size

    entries = _managed_diagnostic_entries(storage_root)
    if cutoff is not None:
        for path, session_id in entries:
            if session_id in protected or not path.exists():
                continue
            if _entry_timestamp(path) < cutoff:
                remove_entry(path)

    entries = [item for item in _managed_diagnostic_entries(storage_root) if item[0].exists()]
    total_bytes = sum(_entry_size(path) for path, _ in entries)
    byte_limit = max(0, int(max_bytes))
    if byte_limit and total_bytes > byte_limit:
        removable = sorted(
            (item for item in entries if item[1] not in protected),
            key=lambda item: (_entry_timestamp(item[0]), str(item[0])),
        )
        for path, _session_id in removable:
            if total_bytes <= byte_limit:
                break
            size = _entry_size(path)
            remove_entry(path)
            total_bytes -= size

    remaining = _managed_diagnostic_entries(storage_root)
    return {
        "removed_entries": removed_entries,
        "removed_bytes": removed_bytes,
        "remaining_entries": len(remaining),
        "remaining_bytes": sum(_entry_size(path) for path, _ in remaining),
    }


@dataclass
class DiagnosticDecision:
    decision_id: int
    actor: int
    actor_kind: str
    observation: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    action: dict[str, Any]
    elapsed_ms: float | None = None
    policy_metadata: dict[str, Any] | None = None
    created_at: str = field(default_factory=_utc_now)
    snapshot: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.elapsed_ms is None:
            payload.pop("elapsed_ms", None)
        if self.policy_metadata is None:
            payload.pop("policy_metadata", None)
        if self.snapshot is None:
            payload.pop("snapshot", None)
        return payload


class DiagnosticSessionRecorder:
    """Append-only local evidence for human-vs-AI diagnosis and relabeling.

    Each private snapshot is a trusted-local pickle.  It is intentionally kept
    outside the JSONL stream so tools can inspect public records without ever
    deserializing executable Python data.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        save_private_snapshots: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or (
            datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        )
        if not self.session_id.replace("-", "").isalnum():
            raise ValueError("session_id must contain only letters, digits, or hyphens")
        self.path = self.root / self.session_id
        self.path.mkdir(parents=False, exist_ok=False)
        self.snapshot_path = self.path / "snapshots"
        self.save_private_snapshots = bool(save_private_snapshots)
        if self.save_private_snapshots:
            self.snapshot_path.mkdir()
        self.decisions_path = self.path / "decisions.jsonl"
        self.markers_path = self.path / "markers.jsonl"
        self._lock = threading.RLock()
        self._next_decision_id = 0
        self._manifest = {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": _utc_now(),
            "finished_at": None,
            "status": "active",
            "decision_count": 0,
            "marker_count": 0,
            "save_private_snapshots": self.save_private_snapshots,
            "metadata": dict(metadata or {}),
        }
        _atomic_json(self.path / "manifest.json", self._manifest)

    @property
    def latest_decision_id(self) -> int | None:
        with self._lock:
            return self._next_decision_id - 1 if self._next_decision_id else None

    def record_decision(
        self,
        *,
        actor: int,
        actor_kind: str,
        observation: dict[str, Any],
        legal_actions: Sequence[Action | dict[str, Any]],
        action: Action | dict[str, Any],
        elapsed_ms: float | None = None,
        policy_metadata: dict[str, Any] | None = None,
        engine=None,
    ) -> DiagnosticDecision:
        if actor_kind not in {"human", "ai", "system"}:
            raise ValueError("actor_kind must be human, ai, or system")
        action_payload = action.to_dict() if isinstance(action, Action) else dict(action)
        legal_payloads = [
            item.to_dict() if isinstance(item, Action) else dict(item)
            for item in legal_actions
        ]
        with self._lock:
            decision_id = self._next_decision_id
            self._next_decision_id += 1
            snapshot_name = None
            if self.save_private_snapshots and engine is not None:
                snapshot_name = f"decision-{decision_id:06d}.pkl.gz"
                snapshot_file = self.snapshot_path / snapshot_name
                temporary = snapshot_file.with_suffix(snapshot_file.suffix + ".tmp")
                with gzip.open(temporary, "wb", compresslevel=5) as handle:
                    pickle.dump(engine, handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, snapshot_file)
            record = DiagnosticDecision(
                decision_id=decision_id,
                actor=int(actor),
                actor_kind=actor_kind,
                observation=dict(observation),
                legal_actions=legal_payloads,
                action=action_payload,
                elapsed_ms=None if elapsed_ms is None else round(float(elapsed_ms), 3),
                policy_metadata=(
                    None if policy_metadata is None else dict(policy_metadata)
                ),
                snapshot=(f"snapshots/{snapshot_name}" if snapshot_name else None),
            )
            with self.decisions_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            self._manifest["decision_count"] = self._next_decision_id
            _atomic_json(self.path / "manifest.json", self._manifest)
            return record

    def mark(
        self,
        decision_id: int | None = None,
        *,
        label: str = "review",
        note: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            if decision_id is None:
                decision_id = self.latest_decision_id
            if decision_id is None or not 0 <= int(decision_id) < self._next_decision_id:
                raise ValueError("decision_id does not exist")
            marker = {
                "decision_id": int(decision_id),
                "label": str(label or "review")[:80],
                "note": str(note or "")[:2000],
                "created_at": _utc_now(),
            }
            with self.markers_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(marker, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            self._manifest["marker_count"] = int(self._manifest["marker_count"]) + 1
            _atomic_json(self.path / "manifest.json", self._manifest)
            return marker

    def finish(self, *, outcome: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._manifest["status"] = "finished"
            self._manifest["finished_at"] = _utc_now()
            if outcome is not None:
                self._manifest["outcome"] = dict(outcome)
            _atomic_json(self.path / "manifest.json", self._manifest)

    def export(self, destination: str | Path | None = None) -> Path:
        with self._lock:
            if destination is None:
                export_dir = self.root / "exports"
                export_dir.mkdir(exist_ok=True)
                destination = export_dir / f"{self.session_id}.gtnai.zip"
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with zipfile.ZipFile(
                temporary,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for source in sorted(self.path.rglob("*")):
                    if source.is_file():
                        archive.write(source, source.relative_to(self.path))
            os.replace(temporary, destination)
            return destination

    def discard(self) -> None:
        with self._lock:
            shutil.rmtree(self.path, ignore_errors=True)


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
