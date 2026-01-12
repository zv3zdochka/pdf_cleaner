#!/usr/bin/env python3
import os
from pathlib import Path

# =========================
# НАСТРОЙКИ (правьте здесь)
# =========================

# 1) Исключать по ИМЕНИ директории на любом уровне:
#    "dist" исключит ./dist, ./frontend/dist, ./a/b/dist и т.д.
SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build",
    ".venv", "venv", "env",
    ".idea", ".vscode",
}

# 2) Исключать по ОТНОСИТЕЛЬНОМУ ПУТИ от корня (точное поддерево):
#    "backend/generated" исключит только это поддерево (и всё внутри).
SKIP_DIR_PATHS = {
    "previous",
    "data",
    "show_up.py",
    ".dockerignore",
    ".gitignore"
    # "backend/generated",
    # "assets/vendor",
}

# Максимум байт, выводимых на файл (чтобы не улетать в гигабайты)
MAX_BYTES_PER_FILE = 1024 * 1024  # 1 MB

# Печатать бинарные файлы (НЕ рекомендуется). Если False — бинарные пропускаются.
INCLUDE_BINARY_AS_HEX = False

# =========================
# КОД (обычно не трогать)
# =========================

def is_binary_bytes(chunk: bytes) -> bool:
    if b"\x00" in chunk:
        return True
    text_like = sum((b == 9 or b == 10 or b == 13 or 32 <= b <= 126) for b in chunk)
    return (len(chunk) > 0) and (text_like / len(chunk) < 0.85)

def normalize_rel_path(p: str) -> Path:
    # нормализуем "a//b/./c" -> Path("a/b/c")
    parts = [x for x in Path(p).parts if x not in ("", ".",)]
    return Path(*parts)

def is_under_skipped_dir(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)

    # 1) по имени директории
    for part in rel.parts:
        if part in SKIP_DIR_NAMES:
            return True

    # 2) по относительному пути (поддерево)
    relp = Path(rel.as_posix())
    for s in SKIP_DIR_PATHS:
        s_norm = normalize_rel_path(s)
        try:
            relp.relative_to(s_norm)
            return True
        except ValueError:
            pass

    return False

def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)

        # фильтруем директории на лету, чтобы os.walk туда не заходил
        kept = []
        for d in dirnames:
            candidate = dp / d
            if d in SKIP_DIR_NAMES:
                continue
            if is_under_skipped_dir(candidate, root):
                continue
            kept.append(d)
        dirnames[:] = kept

        for fn in filenames:
            p = dp / fn
            if is_under_skipped_dir(p, root):
                continue
            if p.is_file():
                files.append(p)

    files.sort(key=lambda p: p.relative_to(root).as_posix())
    return files

def read_text_safely(path: Path, max_bytes: int) -> tuple[str | None, str | None]:
    """
    Возвращает (text, reason_skipped).
    Если файл бинарный или не читается — text=None, reason_skipped=str.
    """
    try:
        with path.open("rb") as f:
            head = f.read(min(4096, max_bytes))
            if is_binary_bytes(head):
                return None, "binary file"
            data = head
            remaining = max_bytes - len(head)
            if remaining > 0:
                data += f.read(remaining)
            return data.decode("utf-8", errors="replace"), None
    except Exception as e:
        return None, f"read error: {e.__class__.__name__}: {e}"

def main() -> None:
    root = Path(".").resolve()

    # быстрый sanity-check: SKIP_DIR_PATHS должны быть относительными
    for s in list(SKIP_DIR_PATHS):
        if Path(s).is_absolute():
            raise ValueError(f"SKIP_DIR_PATHS must be relative, got absolute: {s}")

    files = iter_files(root)

    for p in files:
        rel = p.relative_to(root).as_posix()
        print("=" * 80)
        print(f"FILE: {rel}")
        print("-" * 80)

        text, skipped = read_text_safely(p, MAX_BYTES_PER_FILE)
        if skipped is None and text is not None:
            print(text, end="" if text.endswith("\n") else "\n")
            size = p.stat().st_size
            if size > MAX_BYTES_PER_FILE:
                print(f"\n[TRUNCATED] file size={size} bytes, shown first {MAX_BYTES_PER_FILE} bytes")
        else:
            if INCLUDE_BINARY_AS_HEX and skipped == "binary file":
                data = p.read_bytes()[:MAX_BYTES_PER_FILE]
                print(f"[BINARY as HEX] first {len(data)} bytes:\n{data.hex()}")
                size = p.stat().st_size
                if size > MAX_BYTES_PER_FILE:
                    print(f"\n[TRUNCATED] file size={size} bytes, shown first {MAX_BYTES_PER_FILE} bytes")
            else:
                print(f"[SKIPPED] {rel}: {skipped}")

        print()

if __name__ == "__main__":
    main()
