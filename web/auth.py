from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


security = HTTPBasic(auto_error=False)


def require_basic_auth(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> None:
    user = os.getenv("WEB_USER", "").strip()
    pwd = os.getenv("WEB_PASSWORD", "").strip()
    if not user or not pwd:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="WEB_USER/WEB_PASSWORD are not configured",
            headers={"WWW-Authenticate": "Basic"},
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    ok_user = secrets.compare_digest(credentials.username, user)
    ok_pwd = secrets.compare_digest(credentials.password, pwd)
    if not (ok_user and ok_pwd):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
