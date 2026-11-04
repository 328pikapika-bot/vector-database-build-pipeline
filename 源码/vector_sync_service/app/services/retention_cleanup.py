"""Retention cleanup for Mongo runtime state and local attachment artifacts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Optional

from app.config import settings
from app.observability.logger import get_logger
from app.services.mongo_state_manager import (
    MongoStateManager,
    get_existing_mongo_state_manager,
    get_mongo_state_manager,
)

logger = get_logger(__name__)

_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_cleanup_scheduler: Optional["RetentionCleanupScheduler"] = None


@dataclass(slots=True)
class CleanupResult:
    mode: str
    dry_run: bool
    started_at: str
    finished_at: str = ""
    mongo: dict[str, int] = field(default_factory=dict)
    files: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dry_run": self.dry_run,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "mongo": dict(self.mongo),
            "files": dict(self.files),
            "errors": list(self.errors),
        }


class RetentionCleanupService:
    def __init__(
        self,
        *,
        manager: Optional[MongoStateManager] = None,
        upload_dir: Optional[Path] = None,
    ) -> None:
        self._manager = manager
        self._upload_dir = upload_dir

    async def run(self, *, mode: str = "daily", dry_run: bool = False) -> dict[str, Any]:
        manager = self._manager or get_existing_mongo_state_manager() or get_mongo_state_manager()
        if not getattr(manager, "connected", False):
            await manager.connect()
        return await asyncio.to_thread(
            self._run_with_manager_sync,
            manager,
            mode,
            dry_run,
        )

    def run_sync(self, *, mode: str = "daily", dry_run: bool = False) -> dict[str, Any]:
        manager = self._manager
        if manager is None:
            manager = get_existing_mongo_state_manager()
        if manager is None or not _manager_has_db(manager):
            raise RuntimeError("retention cleanup requires a connected mongo state manager")
        return self._run_with_manager_sync(manager, mode, dry_run)

    def _run_with_manager_sync(
        self,
        manager: MongoStateManager,
        mode: str,
        dry_run: bool,
    ) -> dict[str, Any]:
        resolved_mode = _cleanup_mode(mode)
        result = CleanupResult(
            mode=resolved_mode,
            dry_run=bool(dry_run),
            started_at=_utcnow().isoformat(),
        )
        try:
            self._cleanup_mongo(
                manager,
                result,
                deep=resolved_mode == "weekly",
                dry_run=dry_run,
            )
        except Exception as error:  # noqa: BLE001 - cleanup must report and continue
            message = f"mongo cleanup failed: {error}"
            logger.warning("retention_cleanup_mongo_failed", error=str(error))
            result.errors.append(message)
        try:
            self._cleanup_files(result, deep=resolved_mode == "weekly", dry_run=dry_run)
        except Exception as error:  # noqa: BLE001 - cleanup must report and continue
            message = f"file cleanup failed: {error}"
            logger.warning("retention_cleanup_files_failed", error=str(error))
            result.errors.append(message)
        result.finished_at = _utcnow().isoformat()
        logger.info("retention_cleanup_finished", **result.to_dict())
        return result.to_dict()

    def _cleanup_mongo(
        self,
        manager: MongoStateManager,
        result: CleanupResult,
        *,
        deep: bool,
        dry_run: bool,
    ) -> None:
        now = _utcnow()
        artifact_filter = {"expire_at": {"$lte": now}}
        result.mongo["expired_artifacts"] = _delete_or_count(
            manager.artifacts_collection,
            artifact_filter,
            dry_run=dry_run,
        )

        if deep:
            success_cutoff = now - timedelta(
                days=max(int(settings.MONGO_SUCCESS_TASK_RETENTION_DAYS or 7), 0)
            )
            success_filter = {
                "status": "SUCCESS",
                "finished_at": {"$lte": success_cutoff},
            }
            result.mongo["success_fulltext_tasks"] = _delete_or_count(
                manager.fulltext_tasks_collection,
                success_filter,
                dry_run=dry_run,
            )
            result.mongo["success_image_tasks"] = _delete_or_count(
                manager.image_tasks_collection,
                success_filter,
                dry_run=dry_run,
            )

    def _cleanup_files(self, result: CleanupResult, *, deep: bool, dry_run: bool) -> None:
        upload_root = self._upload_root()
        now = _utcnow()
        success_cutoff = now - timedelta(
            hours=max(int(settings.IMAGE_LOCAL_SUCCESS_RETENTION_HOURS or 24), 0)
        )
        failure_cutoff = now - timedelta(
            days=max(int(settings.LOCAL_FAILED_RETENTION_DAYS or 15), 0)
        )
        report_cutoff = now - timedelta(days=max(int(settings.RETENTION_CLEANUP_REPORT_RETENTION_DAYS or 7), 0))

        attachment_result = self._cleanup_success_attachments(
            result=result,
            upload_root=upload_root,
            success_cutoff=success_cutoff,
            dry_run=dry_run,
        )
        result.files.update(attachment_result)
        result.files["out_dirs"] = _delete_old_children(
            _SERVICE_ROOT / "out",
            report_cutoff,
            dry_run=dry_run,
            prefix_filter=("codex_", "clean_", "tmp_", "retention_"),
        )
        result.files["failed_dirs"] = _delete_old_children(
            upload_root / "failed",
            failure_cutoff,
            dry_run=dry_run,
        )
        result.files["empty_dirs"] = _remove_empty_dirs(
            upload_root,
            dry_run=dry_run,
        )

    def _upload_root(self) -> Path:
        if self._upload_dir is not None:
            return self._upload_dir
        configured = Path(str(settings.UPLOAD_DIR or "upload"))
        if configured.is_absolute():
            return configured
        return _SERVICE_ROOT / configured

    def _cleanup_success_attachments(
        self,
        *,
        result: CleanupResult,
        upload_root: Path,
        success_cutoff: datetime,
        dry_run: bool,
    ) -> dict[str, int]:
        attachments_root = upload_root / "attachments"
        if not attachments_root.exists() or not attachments_root.is_dir():
            return {
                "attachments_scanned": 0,
                "attachments_confirmed_success": 0,
                "attachments_deleted": 0,
                "attachments_deleted_images": 0,
                "attachments_deleted_mmd": 0,
                "attachments_deleted_bytes": 0,
                "attachments_skipped": 0,
            }

        stats = {
            "attachments_scanned": 0,
            "attachments_confirmed_success": 0,
            "attachments_deleted": 0,
            "attachments_deleted_images": 0,
            "attachments_deleted_mmd": 0,
            "attachments_deleted_bytes": 0,
            "attachments_skipped": 0,
        }
        skip_reasons: dict[str, int] = {}
        for task_dir in _iter_task_dirs(attachments_root):
            stats["attachments_scanned"] += 1
            manifest = _load_manifest(task_dir / "manifest.json")
            if manifest is None:
                _bump_skip(skip_reasons, "missing_manifest")
                stats["attachments_skipped"] += 1
                continue
            task_id = str(manifest.get("task_id") or "").strip()
            if not task_id:
                _bump_skip(skip_reasons, "missing_task_id")
                stats["attachments_skipped"] += 1
                continue
            task_doc = self._lookup_image_task(task_id)
            if task_doc is None:
                _bump_skip(skip_reasons, "task_not_found")
                stats["attachments_skipped"] += 1
                continue
            task_status = str(task_doc.get("status") or "").strip().upper()
            if task_status != "SUCCESS":
                _bump_skip(skip_reasons, f"status_{task_status or 'unknown'}")
                stats["attachments_skipped"] += 1
                continue
            finished_at = self._resolve_task_finished_at(task_doc, manifest)
            if finished_at is None:
                _bump_skip(skip_reasons, "invalid_manifest_time")
                stats["attachments_skipped"] += 1
                continue
            if finished_at > success_cutoff:
                _bump_skip(skip_reasons, "within_retention_window")
                stats["attachments_skipped"] += 1
                continue

            stats["attachments_confirmed_success"] += 1
            deleted_images, deleted_mmd, deleted_bytes = _count_attachment_payloads(task_dir)
            if not dry_run:
                shutil.rmtree(task_dir, ignore_errors=True)
            stats["attachments_deleted"] += 1
            stats["attachments_deleted_images"] += deleted_images
            stats["attachments_deleted_mmd"] += deleted_mmd
            stats["attachments_deleted_bytes"] += deleted_bytes

        for reason, count in skip_reasons.items():
            result.files[f"attachments_skip_{reason}"] = count
        return stats

    def _lookup_image_task(self, task_id: str) -> Optional[dict[str, Any]]:
        manager = self._manager or get_existing_mongo_state_manager()
        if manager is None or not _manager_has_db(manager):
            return None
        try:
            return manager.image_tasks_collection.find_one({"_id": task_id})
        except Exception:
            return None

    def _resolve_task_finished_at(
        self,
        task_doc: dict[str, Any],
        manifest: dict[str, Any],
    ) -> Optional[datetime]:
        for key in ("finished_at", "updated_at", "created_at"):
            value = task_doc.get(key)
            if value:
                parsed = _coerce_datetime(value)
                if parsed is not None:
                    return parsed
        return _parse_manifest_time(manifest)


class RetentionCleanupScheduler:
    def __init__(self, service: Optional[RetentionCleanupService] = None) -> None:
        self._service = service or RetentionCleanupService()
        self._task: Optional[asyncio.Task[None]] = None
        self._last_run: Optional[dict[str, Any]] = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self.run_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def run_loop(self) -> None:
        interval_seconds = max(
            int(settings.RETENTION_CLEANUP_DAILY_INTERVAL_HOURS or 24) * 3600,
            300,
        )
        while True:
            try:
                mode = "weekly" if _utcnow().weekday() == int(settings.RETENTION_CLEANUP_WEEKLY_DAY or 0) else "daily"
                self._last_run = await self._service.run(mode=mode, dry_run=False)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - scheduler should survive cleanup errors
                logger.warning("retention_cleanup_scheduler_failed", error=str(error))
                self._last_run = {
                    "mode": "unknown",
                    "dry_run": False,
                    "errors": [str(error)],
                    "finished_at": _utcnow().isoformat(),
                }
            await asyncio.sleep(interval_seconds)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(settings.RETENTION_CLEANUP_ENABLED),
            "running": self._task is not None and not self._task.done(),
            "daily_interval_hours": settings.RETENTION_CLEANUP_DAILY_INTERVAL_HOURS,
            "weekly_day": settings.RETENTION_CLEANUP_WEEKLY_DAY,
            "last_run": self._last_run,
        }


def get_retention_cleanup_scheduler() -> RetentionCleanupScheduler:
    global _cleanup_scheduler
    if _cleanup_scheduler is None:
        _cleanup_scheduler = RetentionCleanupScheduler()
    return _cleanup_scheduler


def _cleanup_mode(value: str) -> str:
    normalized = str(value or "daily").strip().lower()
    return "weekly" if normalized in {"weekly", "week", "deep"} else "daily"


def _delete_or_count(collection: Any, filter_doc: dict[str, Any], *, dry_run: bool) -> int:
    if dry_run:
        return int(collection.count_documents(filter_doc))
    result = collection.delete_many(filter_doc)
    return int(getattr(result, "deleted_count", 0) or 0)


def _manager_has_db(manager: MongoStateManager) -> bool:
    return bool(
        getattr(manager, "connected", False)
        or (
            getattr(manager, "_client", None) is not None
            and getattr(manager, "_db", None) is not None
        )
    )


def _delete_old_children(
    root: Path,
    cutoff: datetime,
    *,
    dry_run: bool,
    prefix_filter: tuple[str, ...] = (),
) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    deleted = 0
    for child in root.iterdir():
        if prefix_filter and not child.name.startswith(prefix_filter):
            continue
        try:
            modified = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if modified > cutoff:
            continue
        deleted += 1
        if dry_run:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    return deleted


def _iter_task_dirs(root: Path):
    for doc_dir in root.iterdir():
        if not doc_dir.is_dir():
            continue
        for task_dir in doc_dir.iterdir():
            if task_dir.is_dir():
                yield task_dir


def _load_manifest(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_manifest_time(manifest: dict[str, Any]) -> Optional[datetime]:
    raw = manifest.get("created_at")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except Exception:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _count_attachment_payloads(task_dir: Path) -> tuple[int, int, int]:
    image_count = 0
    mmd_count = 0
    total_bytes = 0
    for child in task_dir.rglob("*"):
        if not child.is_file():
            continue
        try:
            total_bytes += int(child.stat().st_size)
        except OSError:
            continue
        suffix = child.suffix.lower()
        if suffix == ".mmd":
            mmd_count += 1
        elif child.name != "manifest.json":
            image_count += 1
    return image_count, mmd_count, total_bytes


def _bump_skip(skip_reasons: dict[str, int], reason: str) -> None:
    skip_reasons[reason] = int(skip_reasons.get(reason, 0)) + 1


def _remove_empty_dirs(root: Path, *, dry_run: bool) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    removed = 0
    for child in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not child.is_dir():
            continue
        try:
            next(child.iterdir())
        except StopIteration:
            removed += 1
            if not dry_run:
                child.rmdir()
        except OSError:
            continue
    return removed


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
