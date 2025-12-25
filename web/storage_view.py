from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n or 0)
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    if i == 0:
        return f"{int(x)} {units[i]}"
    return f"{x:.2f} {units[i]}"


def ts_h(ts: Optional[int]) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


@dataclass
class FileItem:
    kind: str
    name: str
    rel_path: str
    exists: bool
    size: int

    @property
    def size_h(self) -> str:
        return human_bytes(self.size)


@dataclass
class RequestRow:
    user_id: int
    request_id: str
    status: Optional[str]
    original_filename: Optional[str]
    created_at: Optional[int]
    total_size: int

    @property
    def created_h(self) -> str:
        return ts_h(self.created_at)

    @property
    def total_size_h(self) -> str:
        return human_bytes(self.total_size)


@dataclass
class RequestDetails(RequestRow):
    meta: Dict[str, Any]
    files: List[FileItem]

    @property
    def meta_pretty(self) -> str:
        try:
            return json.dumps(self.meta, ensure_ascii=False, indent=2)
        except Exception:
            return str(self.meta)


class StorageView:
    def __init__(self, storage_root: Path):
        self.root = storage_root

    def _dir_size(self, p: Path) -> int:
        total = 0
        for root, _, files in os.walk(p):
            for fn in files:
                try:
                    total += (Path(root) / fn).stat().st_size
                except FileNotFoundError:
                    continue
        return total

    def _meta(self, req_dir: Path) -> Dict[str, Any]:
        mp = req_dir / "meta.json"
        if not mp.exists():
            return {}
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _iter_req_dirs(self) -> Iterable[Tuple[int, str, Path]]:
        users_dir = self.root / "users"
        if not users_dir.exists():
            return []
        for user_dir in users_dir.iterdir():
            if not user_dir.is_dir():
                continue
            try:
                uid = int(user_dir.name)
            except Exception:
                continue
            reqs = user_dir / "requests"
            if not reqs.exists():
                continue
            for req_dir in reqs.iterdir():
                if req_dir.is_dir():
                    yield uid, req_dir.name, req_dir

    def stats(self) -> Tuple[int, int]:
        return self._dir_size(self.root), self._count_users()

    def _count_users(self) -> int:
        users_dir = self.root / "users"
        if not users_dir.exists():
            return 0
        return sum(1 for p in users_dir.iterdir() if p.is_dir())

    def list_requests(self, q: Optional[str] = None, limit: int = 2000) -> List[RequestRow]:
        qn = (q or "").strip().lower()
        rows: List[RequestRow] = []
        for uid, rid, req_dir in self._iter_req_dirs():
            meta = self._meta(req_dir)
            status = meta.get("status")
            original = meta.get("original_filename")
            created = meta.get("created_at")
            total = self._dir_size(req_dir)

            row = RequestRow(
                user_id=uid,
                request_id=rid,
                status=status,
                original_filename=original,
                created_at=created,
                total_size=total,
            )

            if qn:
                hay = f"{uid} {rid} {status or ''} {original or ''}".lower()
                if qn not in hay:
                    continue

            rows.append(row)
            if len(rows) >= limit:
                break

        rows.sort(key=lambda r: (r.created_at or 0, r.request_id), reverse=True)
        return rows

    def get_request(self, user_id: int, request_id: str) -> Optional[RequestDetails]:
        req_dir = self.root / "users" / str(user_id) / "requests" / request_id
        if not req_dir.exists():
            return None

        meta = self._meta(req_dir)
        status = meta.get("status")
        original = meta.get("original_filename")
        created = meta.get("created_at")
        total = self._dir_size(req_dir)

        def fi(kind: str, name: str) -> FileItem:
            p = req_dir / name
            exists = p.exists()
            size = p.stat().st_size if exists else 0
            rel = str(p.relative_to(self.root)) if exists else str((req_dir / name).relative_to(self.root))
            return FileItem(kind=kind, name=name, rel_path=rel, exists=exists, size=size)

        files = [
            fi("input", "input.pdf"),
            fi("cleaned", "cleaned.pdf"),
            fi("cleaned_small", "cleaned_small.pdf"),
            fi("meta", "meta.json"),
        ]

        return RequestDetails(
            user_id=user_id,
            request_id=request_id,
            status=status,
            original_filename=original,
            created_at=created,
            total_size=total,
            meta=meta,
            files=files,
        )

    def delete_request(self, user_id: int, request_id: str) -> bool:
        req_dir = self.root / "users" / str(user_id) / "requests" / request_id
        if not req_dir.exists():
            return False
        import shutil
        shutil.rmtree(req_dir, ignore_errors=True)
        try:
            reqs_dir = req_dir.parent
            user_dir = reqs_dir.parent
            if reqs_dir.exists() and not any(reqs_dir.iterdir()):
                reqs_dir.rmdir()
            if user_dir.exists() and not any(user_dir.iterdir()):
                user_dir.rmdir()
        except Exception:
            pass
        return True

    def file_path(self, user_id: int, request_id: str, filename: str) -> Path:
        allowed = {"input.pdf", "cleaned.pdf", "cleaned_small.pdf", "meta.json"}
        if filename not in allowed:
            raise ValueError("File not allowed")
        p = self.root / "users" / str(user_id) / "requests" / request_id / filename
        rp = p.resolve()
        rr = self.root.resolve()
        if rr not in rp.parents and rp != rr:
            raise ValueError("Unsafe path")
        return p
