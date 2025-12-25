"""PDF compression utilities."""

from __future__ import annotations

from pathlib import Path

import pikepdf


def shrink_pdf(input_path: str | Path, output_path: str | Path) -> None:
    """Compress a PDF using pikepdf.

    This step does not change PDF semantics; it primarily reduces size by:
    - compressing streams
    - packing objects into object streams

    Parameters
    ----------
    input_path:
        Source PDF.
    output_path:
        Destination PDF.
    """
    input_path = str(input_path)
    output_path = str(output_path)
    with pikepdf.Pdf.open(input_path) as pdf:
        pdf.save(
            output_path,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )
