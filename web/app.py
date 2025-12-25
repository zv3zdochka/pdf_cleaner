from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from web.auth import require_basic_auth
from web.storage_view import StorageView, human_bytes


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
STORAGE_MAX_BYTES = _env_int("STORAGE_MAX_BYTES", 30 * 1024 * 1024 * 1024)  # 30 GiB

app = FastAPI(title="PDF Cleaner Web")

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

sv = StorageView(STORAGE_DIR)


def _layout_context(q: str | None = None):
    used, users = sv.stats()
    rows = sv.list_requests(q=q, limit=2000)
    return {
        "used_h": human_bytes(used),
        "limit_h": human_bytes(STORAGE_MAX_BYTES),
        "total_requests": len(rows),
        "total_users": users,
    }


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_basic_auth)])
def index(request: Request, q: str | None = None):
    rows = sv.list_requests(q=q, limit=2000)
    ctx = _layout_context(q=q)
    ctx.update({
        "request": request,
        "title": "PDF Cleaner — Requests",
        "rows": rows,
        "q": q,
    })
    return templates.TemplateResponse("index.html", ctx)


@app.get("/users/{user_id}/requests/{request_id}", response_class=HTMLResponse, dependencies=[Depends(require_basic_auth)])
def request_details(request: Request, user_id: int, request_id: str):
    req = sv.get_request(user_id, request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Not found")
    ctx = _layout_context()
    ctx.update({
        "request": request,
        "title": f"Request {request_id}",
        "req": req,
    })
    return templates.TemplateResponse("request.html", ctx)


@app.get("/download/{user_id}/{request_id}/{filename}", dependencies=[Depends(require_basic_auth)])
def download(user_id: int, request_id: str, filename: str):
    try:
        p = sv.file_path(user_id, request_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(p), filename=filename)


@app.delete("/api/users/{user_id}/requests/{request_id}", response_class=PlainTextResponse, dependencies=[Depends(require_basic_auth)])
def api_delete_request(user_id: int, request_id: str):
    ok = sv.delete_request(user_id, request_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return "ok"
