"""Top-level package for the PDF Cleaner bot.

This package contains:
- a Telegram bot (aiogram) entrypoint and handlers;
- PDF processing services (PyMuPDF redaction + ONNX detector);
- small utilities (PDF compression, logging setup).

Public API is intentionally small to keep future extensions (e.g., Web API) clean.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
