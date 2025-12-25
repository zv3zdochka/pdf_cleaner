from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, Tuple


@dataclass(frozen=True)
class StorageConfig:
    root_dir: Path
    max_bytes: int
    max_age_days: int


class StorageManager:
    """
    File storage for incoming and processed PDFs.

    Layout (web-friendly):
      {root}/users/{user_id}/requests/{request_id}/
        - input.pdf
        - cleaned.pdf
        - cleaned_small.pdf
        - meta.json

    Retention policy:
      - delete requests older than max_age_days
      - enforce total storage size <= max_bytes by deleting oldest requests
    """

    def __init__(self, cfg: StorageConfig, logger):
        self.cfg = cfg
        self.log = logger
        self.cfg.root_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Paths & metadata
    # ----------------------------

    def request_dir(self, user_id: int, request_id: str) -> Path:
        return self.cfg.root_dir / "users" / str(user_id) / "requests" / request_id

    def meta_path(self, user_id: int, request_id: str) -> Path:
        return self.request_dir(user_id, request_id) / "meta.json"

    def write_meta(self, user_id: int, request_id: str, meta: Dict[str, Any]) -> None:
        rd = self.request_dir(user_id, request_id)
        rd.mkdir(parents=True, exist_ok=True)
        tmp = rd / "meta.json.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        tmp.replace(rd / "meta.json")

    def read_meta(self, user_id: int, request_id: str) -> Optional[Dict[str, Any]]:
        mp = self.meta_path(user_id, request_id)
        if not mp.exists():
            return None
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_request_dirs(self) -> Iterable[Path]:
        users_dir = self.cfg.root_dir / "users"
        if not users_dir.exists():
            return []
        # users/<uid>/requests/<rid>
        for user_dir in users_dir.iterdir():
            reqs = user_dir / "requests"
            if not reqs.exists():
                continue
            for req_dir in reqs.iterdir():
                if req_dir.is_dir():
                    yield req_dir

    # ----------------------------
    # Retention / quota
    # ----------------------------

    def _dir_size_bytes(self, p: Path) -> int:
        total = 0
        for root, _, files in os.walk(p):
            for fn in files:
                try:
                    total += (Path(root) / fn).stat().st_size
                except FileNotFoundError:
                    continue
        return total

    def total_size_bytes(self) -> int:
        return self._dir_size_bytes(self.cfg.root_dir)

    def _request_mtime(self, req_dir: Path) -> float:
        """
        Determine request "age" by meta.json mtime (preferred) else directory mtime.
        """
        mp = req_dir / "meta.json"
        try:
            return mp.stat().st_mtime
        except FileNotFoundError:
            try:
                return req_dir.stat().st_mtime
            except FileNotFoundError:
                return time.time()

    def cleanup(self) -> None:
        """
        Retention is disabled by design (manual deletion via web UI).
        If max_age_days > 0 you can optionally enable TTL cleanup, but
        quota cleanup (auto-delete) is intentionally NOT performed.
        """
        self.cfg.root_dir.mkdir(parents=True, exist_ok=True)

        # TTL optional (only if max_age_days > 0)
        max_age_sec = max(0, self.cfg.max_age_days) * 86400
        if max_age_sec <= 0:
            return

        now = time.time()
        for req_dir in list(self.list_request_dirs()):
            age = now - self._request_mtime(req_dir)
            if age > max_age_sec:
                self._delete_request_dir(req_dir, reason="ttl_expired")

    def _delete_request_dir(self, req_dir: Path, reason: str) -> None:
        try:
            self.log.info("Storage cleanup: deleting %s (reason=%s)", req_dir, reason)
            shutil.rmtree(req_dir, ignore_errors=True)
            self._prune_empty_parents(req_dir)
        except Exception as e:
            self.log.warning("Storage cleanup failed for %s: %s", req_dir, e)

    def would_exceed_quota(self, add_bytes: int = 0) -> bool:
        if self.cfg.max_bytes <= 0:
            return False
        add = max(0, int(add_bytes))
        return (self.total_size_bytes() + add) > self.cfg.max_bytes

    def _prune_empty_parents(self, req_dir: Path) -> None:
        """
        Remove empty parent folders up to root/users/<uid>/requests
        """
        p = req_dir
        for _ in range(4):  # request -> requests -> uid -> users
            p = p.parent
            if p == self.cfg.root_dir:
                return
            try:
                if p.exists() and p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
            except Exception:
                return
