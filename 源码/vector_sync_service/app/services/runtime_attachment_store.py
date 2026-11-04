"""Runtime storage for downloaded .mmd and image static resources."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Optional

from app.config import settings


_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(slots=True)
class RuntimeAttachmentBatch:
    """Per-image runtime folder for downloaded image and .mmd resources."""

    doc_id: str
    batch_id: str
    task_id: str
    root_dir: Path
    failed_root_dir: Path
    manifest: dict[str, Any] = dataclass_field(default_factory=dict)
    mode: str = "full"
    _image_payload: Optional[tuple[str, bytes]] = None
    _mmd_payload: Optional[tuple[str, str]] = None

    @classmethod
    def create(
        cls,
        *,
        doc_id: str,
        batch_id: str,
        task_id: str,
        image_relative_url: str,
        mmd_relative_url: Optional[str] = None,
    ) -> "RuntimeAttachmentBatch":
        root = _resolve_upload_root()
        safe_doc_id = _safe_component(doc_id or "unknown-doc")
        safe_batch_id = _safe_component(batch_id or task_id or "unknown-batch")
        root_dir = root / "attachments" / safe_doc_id / safe_batch_id
        failed_root_dir = root / "failed" / safe_doc_id / safe_batch_id
        mode = _runtime_attachment_mode()
        if mode == "full":
            root_dir.mkdir(parents=True, exist_ok=True)
        batch = cls(
            doc_id=doc_id,
            batch_id=safe_batch_id,
            task_id=task_id,
            root_dir=root_dir,
            failed_root_dir=failed_root_dir,
            mode=mode,
            manifest={
                "doc_id": doc_id,
                "batch_id": safe_batch_id,
                "task_id": task_id,
                "created_at": _utcnow_iso(),
                "image_relative_url": image_relative_url,
                "mmd_relative_url": mmd_relative_url,
                "downloads": {},
                "context": {},
                "failure": None,
            },
        )
        if mode == "full":
            batch.write_manifest()
        return batch

    def record_image_download(
        self,
        *,
        relative_url: str,
        url: str,
        content: bytes,
        content_type: str,
        size_bytes: int,
    ) -> Path:
        filename = _safe_filename(relative_url, fallback="image.bin")
        path = self.root_dir / filename
        if self.mode == "none":
            local_path = ""
        elif self.mode == "failure_only":
            self._image_payload = (filename, bytes(content))
            local_path = str(path)
        else:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            local_path = str(path)
        self.manifest["downloads"]["image"] = {
            "relative_url": relative_url,
            "url": url,
            "local_path": local_path,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "sha256": "" if self.mode == "none" else sha256(content).hexdigest(),
        }
        if self.mode == "full":
            self.write_manifest()
        return path

    def record_mmd_download(
        self,
        *,
        relative_url: str,
        url: str,
        text: str,
        char_count: int,
    ) -> Path:
        filename = _safe_filename(relative_url, fallback="document.mmd")
        path = self.root_dir / filename
        if self.mode == "none":
            local_path = ""
        elif self.mode == "failure_only":
            self._mmd_payload = (filename, str(text or ""))
            local_path = str(path)
        else:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            local_path = str(path)
        self.manifest["downloads"]["mmd"] = {
            "relative_url": relative_url,
            "url": url,
            "local_path": local_path,
            "char_count": char_count,
            "sha256": "" if self.mode == "none" else sha256(text.encode("utf-8")).hexdigest(),
        }
        if self.mode == "full":
            self.write_manifest()
        return path

    def record_context(self, **values: Any) -> None:
        self.manifest["context"].update(values)
        if self.mode == "full":
            self.write_manifest()

    def mark_failure(self, *, error_code: str, error_message: str) -> Path:
        self.manifest["failure"] = {
            "error_code": error_code,
            "error_message": error_message,
            "failed_at": _utcnow_iso(),
        }
        if self.mode == "none":
            return self.failed_root_dir
        if self.mode == "failure_only":
            return self.write_failed()
        self.write_manifest()
        return self.move_to_failed()

    def cleanup_success(self) -> None:
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir)

    def move_to_failed(self) -> Path:
        if not self.root_dir.exists():
            return self.failed_root_dir
        target = _unique_failed_dir(self.failed_root_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.root_dir), str(target))
        return target

    def write_failed(self) -> Path:
        target = _unique_failed_dir(self.failed_root_dir)
        target.mkdir(parents=True, exist_ok=True)
        if self._image_payload is not None:
            filename, content = self._image_payload
            (target / filename).write_bytes(content)
            self.manifest["downloads"]["image"]["local_path"] = str(target / filename)
        if self._mmd_payload is not None:
            filename, text = self._mmd_payload
            (target / filename).write_text(text, encoding="utf-8")
            self.manifest["downloads"]["mmd"]["local_path"] = str(target / filename)
        manifest_path = target / "manifest.json"
        manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return target

    def write_manifest(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def _resolve_upload_root() -> Path:
    configured = Path(str(settings.UPLOAD_DIR or "upload"))
    if configured.is_absolute():
        return configured
    return _SERVICE_ROOT / configured


def _runtime_attachment_mode() -> str:
    value = str(settings.IMAGE_RUNTIME_ATTACHMENT_MODE or "full").strip().lower()
    if value in {"none", "off", "disabled", "disable"}:
        return "none"
    if value in {"failure_only", "failed_only"}:
        return "failure_only"
    return "full"


def _safe_component(value: str) -> str:
    text = _SAFE_COMPONENT_RE.sub("_", str(value or "").strip()).strip("._")
    if text:
        return text[:160]
    return sha256(str(value or "").encode("utf-8")).hexdigest()


def _safe_filename(relative_url: str, *, fallback: str) -> str:
    path = str(relative_url or "").split("?", 1)[0].split("#", 1)[0].replace("\\", "/")
    filename = PurePosixPath(path).name
    safe = _safe_component(filename)
    return safe or fallback


def _unique_failed_dir(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    candidate = path.with_name(f"{path.name}-{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}-{timestamp}-{suffix}")
        suffix += 1
    return candidate


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
