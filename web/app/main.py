from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Setup logging for web app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_cleaner.web")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


@dataclass(frozen=True)
class WebConfig:
    storage_dir: Path
    token: str
    max_bytes: int


def _cfg() -> WebConfig:
    return WebConfig(
        storage_dir=Path(os.getenv("STORAGE_DIR", "/app/storage")),
        token=os.getenv("WEB_TOKEN", "").strip(),
        max_bytes=_env_int("STORAGE_MAX_BYTES", 30 * 1024 * 1024 * 1024),
    )


def _safe_rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def _dir_size_bytes(p: Path) -> int:
    total = 0
    for x in p.rglob("*"):
        if x.is_file():
            try:
                total += x.stat().st_size
            except FileNotFoundError:
                pass
    return total


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _human_bytes(n: int) -> str:
    if n < 0:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    if i == 0:
        return f"{int(v)} {units[i]}"
    return f"{v:.2f} {units[i]}"


def _format_iso_datetime(iso_str: Optional[str]) -> str:
    """Convert ISO datetime string to human-readable format."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_str


def scan_requests(storage_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    users_dir = storage_dir / "users"
    if not users_dir.exists():
        return out

    for user_dir in users_dir.iterdir():
        if not user_dir.is_dir():
            continue
        uid = user_dir.name
        reqs_dir = user_dir / "requests"
        if not reqs_dir.exists():
            continue
        for req_dir in reqs_dir.iterdir():
            if not req_dir.is_dir():
                continue
            rid = req_dir.name
            meta_path = req_dir / "meta.json"
            meta = _read_json(meta_path) or {}

            created_at = int(meta.get("created_at") or 0)
            try:
                mtime = int(meta_path.stat().st_mtime)
            except FileNotFoundError:
                mtime = int(req_dir.stat().st_mtime)

            # Bot storage evolved:
            #  - new: input_original.pdf (+ optional input_trimmed.pdf)
            #  - old: input.pdf
            input_original_path = req_dir / "input_original.pdf"
            legacy_input_path = req_dir / "input.pdf"
            input_path = input_original_path if input_original_path.exists() else legacy_input_path

            trimmed_path = req_dir / "input_trimmed.pdf"
            cleaned_path = req_dir / "cleaned.pdf"
            small_path = req_dir / "cleaned_small.pdf"

            def file_info(p: Path) -> Dict[str, Any]:
                if not p.exists():
                    return {"exists": False, "path": _safe_rel(p, storage_dir), "size_bytes": -1, "size_h": "-"}
                try:
                    sz = p.stat().st_size
                except FileNotFoundError:
                    sz = -1
                return {"exists": True, "path": _safe_rel(p, storage_dir), "size_bytes": sz, "size_h": _human_bytes(sz)}

            original = meta.get("original_filename") or "document.pdf"
            status = meta.get("status") or "unknown"

            out.append(
                {
                    "user_id": int(meta.get("user_id") or uid) if str(meta.get("user_id") or "").isdigit() else uid,
                    "request_id": meta.get("request_id") or rid,
                    "original_filename": original,
                    "status": status,
                    "created_at": created_at,
                    "created_at_h": time.strftime("%Y-%m-%d %H:%M:%S",
                                                  time.localtime(created_at)) if created_at else "-",
                    "mtime": mtime,
                    "mtime_h": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                    "meta_rel": _safe_rel(meta_path, storage_dir),
                    "files": {
                        "input": file_info(input_path),
                        "trimmed": file_info(trimmed_path),
                        "cleaned": file_info(cleaned_path),
                        "cleaned_small": file_info(small_path),
                    },
                }
            )

    out.sort(key=lambda x: (x["created_at"] or 0, x["mtime"]), reverse=True)
    return out


def get_users_from_db(storage_dir: Path) -> List[Dict[str, Any]]:
    """Read users from the SQLite database."""
    db_path = storage_dir / "users.db"

    logger.info("Looking for users database at: %s", db_path)

    if not db_path.exists():
        logger.warning("Users database does not exist: %s", db_path)
        return []

    logger.info("Users database found, size: %d bytes", db_path.stat().st_size)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            logger.warning("Users table does not exist in database")
            conn.close()
            return []

        cursor.execute('''
                       SELECT user_id,
                              username,
                              first_name,
                              last_name,
                              language_code,
                              is_bot,
                              phone_number,
                              first_seen_at,
                              last_activity_at,
                              pdfs_sent,
                              messages_count
                       FROM users
                       ORDER BY last_activity_at DESC
                       ''')

        users = []
        for row in cursor.fetchall():
            user = dict(row)
            user["is_bot"] = bool(user["is_bot"])
            user["first_seen_at_h"] = _format_iso_datetime(user.get("first_seen_at"))
            user["last_activity_at_h"] = _format_iso_datetime(user.get("last_activity_at"))

            # Get associated chats
            cursor.execute('''
                           SELECT chat_id, chat_type, first_seen_at, last_seen_at
                           FROM user_chats
                           WHERE user_id = ?
                           ''', (user["user_id"],))
            user["chats"] = [dict(c) for c in cursor.fetchall()]

            users.append(user)

        conn.close()

        logger.info("Found %d users in database", len(users))
        return users

    except Exception as e:
        logger.error("Error reading users database: %s", e, exc_info=True)
        return []


def get_user_stats_from_db(storage_dir: Path) -> Dict[str, Any]:
    """Get aggregate user statistics from the database."""
    db_path = storage_dir / "users.db"
    default_stats = {"total_users": 0, "total_pdfs": 0, "total_messages": 0, "total_chats": 0}

    if not db_path.exists():
        return default_stats

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            conn.close()
            return default_stats

        cursor.execute("SELECT COUNT(*) as total_users FROM users")
        total_users = cursor.fetchone()["total_users"]

        cursor.execute("SELECT SUM(pdfs_sent) as total_pdfs FROM users")
        row = cursor.fetchone()
        total_pdfs = row["total_pdfs"] if row["total_pdfs"] else 0

        cursor.execute("SELECT SUM(messages_count) as total_messages FROM users")
        row = cursor.fetchone()
        total_messages = row["total_messages"] if row["total_messages"] else 0

        cursor.execute("SELECT COUNT(DISTINCT chat_id) as total_chats FROM user_chats")
        total_chats = cursor.fetchone()["total_chats"]

        conn.close()

        return {
            "total_users": total_users,
            "total_pdfs": total_pdfs,
            "total_messages": total_messages,
            "total_chats": total_chats,
        }

    except Exception as e:
        logger.error("Error getting user stats: %s", e)
        return default_stats


templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app = FastAPI(title="PDF Cleaner Storage", version="1.0.0")


@app.middleware("http")
async def token_guard(request: Request, call_next):
    cfg = _cfg()
    if not cfg.token:
        return await call_next(request)

    provided = request.headers.get("X-Admin-Token", "").strip()
    if not provided:
        provided = (request.cookies.get("admin_token") or "").strip()

    qp_token = request.query_params.get("token", "").strip()
    if (not provided) and qp_token:
        provided = qp_token

    if provided != cfg.token:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    response = await call_next(request)

    if qp_token and qp_token == cfg.token:
        try:
            response.set_cookie(
                key="admin_token",
                value=cfg.token,
                httponly=True,
                samesite="lax",
            )
        except Exception:
            pass

    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, q: str = ""):
    cfg = _cfg()
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)

    reqs = scan_requests(cfg.storage_dir)
    if q:
        ql = q.lower().strip()
        reqs = [
            r for r in reqs
            if ql in str(r["user_id"]).lower()
               or ql in str(r["request_id"]).lower()
               or ql in (r["original_filename"] or "").lower()
               or ql in (r["status"] or "").lower()
        ]

    usage = _dir_size_bytes(cfg.storage_dir)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "items": reqs,
            "q": q,
            "usage_bytes": usage,
            "usage_h": _human_bytes(usage),
            "limit_bytes": cfg.max_bytes,
            "limit_h": _human_bytes(cfg.max_bytes),
        },
    )


@app.get("/users", response_class=HTMLResponse)
async def users_page(request: Request, q: str = ""):
    """Display users page with Telegram user data and statistics."""
    cfg = _cfg()
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)

    # Debug: log database path
    db_path = cfg.storage_dir / "users.db"
    logger.info("Users page requested. DB path: %s, exists: %s", db_path, db_path.exists())

    users = get_users_from_db(cfg.storage_dir)
    stats = get_user_stats_from_db(cfg.storage_dir)

    if q:
        ql = q.lower().strip()
        users = [
            u for u in users
            if ql in str(u["user_id"]).lower()
               or ql in (u["username"] or "").lower()
               or ql in (u["first_name"] or "").lower()
               or ql in (u["last_name"] or "").lower()
        ]

    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "users": users,
            "stats": stats,
            "q": q,
            "db_path": str(db_path),
            "db_exists": db_path.exists(),
        },
    )


@app.get("/api/users")
async def api_users(q: str = ""):
    """API endpoint to get all users."""
    cfg = _cfg()
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)

    users = get_users_from_db(cfg.storage_dir)

    if q:
        ql = q.lower().strip()
        users = [
            u for u in users
            if ql in str(u["user_id"]).lower()
               or ql in (u["username"] or "").lower()
               or ql in (u["first_name"] or "").lower()
               or ql in (u["last_name"] or "").lower()
        ]

    return {"users": users}


@app.get("/api/users/stats")
async def api_users_stats():
    """API endpoint to get user statistics summary."""
    cfg = _cfg()
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    return get_user_stats_from_db(cfg.storage_dir)


@app.get("/api/debug/db")
async def api_debug_db():
    """Debug endpoint to check database status."""
    cfg = _cfg()
    db_path = cfg.storage_dir / "users.db"

    result = {
        "storage_dir": str(cfg.storage_dir),
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "db_size": db_path.stat().st_size if db_path.exists() else 0,
    }

    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            result["tables"] = [row[0] for row in cursor.fetchall()]

            cursor.execute("SELECT COUNT(*) FROM users")
            result["user_count"] = cursor.fetchone()[0]

            conn.close()
        except Exception as e:
            result["error"] = str(e)

    return result


@app.get("/api/requests")
async def api_requests(q: str = ""):
    cfg = _cfg()
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    reqs = scan_requests(cfg.storage_dir)
    if q:
        ql = q.lower().strip()
        reqs = [
            r for r in reqs
            if ql in str(r["user_id"]).lower()
               or ql in str(r["request_id"]).lower()
               or ql in (r["original_filename"] or "").lower()
               or ql in (r["status"] or "").lower()
        ]
    return {"items": reqs}


@app.get("/api/stats")
async def api_stats():
    cfg = _cfg()
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    usage = _dir_size_bytes(cfg.storage_dir)
    return {
        "usage_bytes": usage,
        "usage_h": _human_bytes(usage),
        "limit_bytes": cfg.max_bytes,
        "limit_h": _human_bytes(cfg.max_bytes),
    }


def _resolve_request_dir(storage_dir: Path, user_id: str, request_id: str) -> Path:
    rd = storage_dir / "users" / user_id / "requests" / request_id
    if not rd.exists() or not rd.is_dir():
        raise HTTPException(status_code=404, detail="Request not found")
    return rd


@app.get("/download/{user_id}/{request_id}/{kind}")
async def download_file(user_id: str, request_id: str, kind: str):
    cfg = _cfg()
    rd = _resolve_request_dir(cfg.storage_dir, user_id, request_id)

    if kind == "input":
        # Prefer new name, fallback to legacy
        p = (rd / "input_original.pdf") if (rd / "input_original.pdf").exists() else (rd / "input.pdf")
        default_name = "input.pdf"
    elif kind == "trimmed":
        p = rd / "input_trimmed.pdf"
        default_name = "input_trimmed.pdf"
    elif kind == "cleaned":
        p = rd / "cleaned.pdf"
        default_name = "cleaned.pdf"
    elif kind == "cleaned_small":
        p = rd / "cleaned_small.pdf"
        default_name = "cleaned_small.pdf"
    elif kind == "meta":
        p = rd / "meta.json"
        default_name = "meta.json"
    else:
        raise HTTPException(status_code=400, detail="Invalid kind")

    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")

    meta = _read_json(rd / "meta.json") or {}
    original = (meta.get("original_filename") or "document.pdf").strip()
    if not original.lower().endswith(".pdf"):
        original = "document.pdf"

    if kind == "input":
        filename = original
    elif kind == "trimmed":
        stem = Path(original).stem
        filename = f"{stem}_trimmed.pdf"
    elif kind == "cleaned":
        stem = Path(original).stem
        filename = f"{stem}_cleaned.pdf"
    elif kind == "cleaned_small":
        stem = Path(original).stem
        filename = f"{stem}_cleaned_small.pdf"
    else:
        filename = default_name

    return FileResponse(path=str(p), filename=filename, media_type="application/octet-stream")


@app.delete("/api/requests/{user_id}/{request_id}")
async def delete_request(user_id: str, request_id: str):
    cfg = _cfg()
    rd = _resolve_request_dir(cfg.storage_dir, user_id, request_id)
    try:
        shutil.rmtree(rd, ignore_errors=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}