"""PDF region detection + redaction pipeline.

Core responsibilities
--------------------
1) Render each PDF page to an image (PyMuPDF / fitz).
2) Detect regions on the rendered image (RF-DETR ONNX).
3) Map detected image-space bounding boxes back to PDF coordinates.
4) Apply redactions to remove underlying content.

This module deliberately contains no Telegram-specific logic. That separation
keeps it reusable for future interfaces (e.g., HTTP API / Web UI).
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from pdf_cleaner_bot.ml.rfdetr_onnx import RFDetrONNX


class PDFRegionProcessor:
    """Detect and redact model-specified regions in a PDF."""

    def __init__(
            self,
            model_path: Path,
            input_size: Tuple[int, int] = (736, 736),
            conf_threshold: float = 0.5,
            render_zoom: float = 2.0,
            providers: Optional[List[str]] = None,
    ) -> None:
        """Create a processor instance.

        Parameters
        ----------
        model_path:
            Path to RF-DETR ONNX model.
        input_size:
            Model input size (width, height).
        conf_threshold:
            Confidence threshold for detections.
        render_zoom:
            Zoom factor for rendering PDF pages to images.
        providers:
            ONNX Runtime execution providers.
        """
        self.render_zoom = render_zoom
        self.logger = logging.getLogger(self.__class__.__name__)

        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Initialize detector once; reuse across requests.
        self.detector = RFDetrONNX(
            model_path=model_path,
            input_size=input_size,
            conf_threshold=conf_threshold,
            providers=providers,
            logger=logging.getLogger("RFDetrONNX"),
        )

    def render_page_to_image(self, page: fitz.Page) -> Tuple[Image.Image, fitz.Matrix]:
        """Render a single PDF page to a PIL Image.

        Returns
        -------
        (image, matrix)
            image: rendered page as PIL Image.
            matrix: transformation used during rendering.
        """
        matrix = fitz.Matrix(self.render_zoom, self.render_zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        return img, matrix

    def convert_bbox_to_pdf_coords(
            self,
            bbox: List[int] | Tuple[int, int, int, int],
            page: fitz.Page,
            render_matrix: fitz.Matrix,
    ) -> fitz.Rect:
        """Convert a detection bbox from rendered image coords to PDF coords.

        The renderer applies page rotation; we undo it to correctly map bboxes.

        Parameters
        ----------
        bbox:
            [x1, y1, x2, y2] in rendered image coordinates.
        page:
            PyMuPDF page object.
        render_matrix:
            Matrix used during rendering.

        Returns
        -------
        fitz.Rect
            Rectangle in PDF coordinate space.
        """
        x1, y1, x2, y2 = map(int, bbox)

        rotation = page.rotation

        # Rendered image size (consistent with the saved temp image).
        pix = page.get_pixmap(matrix=render_matrix, alpha=False)
        img_width, img_height = pix.width, pix.height

        # Rotation correction: undo rotation applied by renderer.
        if rotation == 90:
            # 90° clockwise: x' = y, y' = width - x
            x1, y1, x2, y2 = y1, img_width - x2, y2, img_width - x1
        elif rotation == 180:
            # 180°: x' = width - x, y' = height - y
            x1, y1, x2, y2 = img_width - x2, img_height - y2, img_width - x1, img_height - y1
        elif rotation == 270:
            # 270° clockwise: x' = height - y, y' = x
            x1, y1, x2, y2 = img_height - y2, x1, img_height - y1, x2

        # Inverse matrix maps from image pixels back to PDF coordinates.
        inv_matrix = ~render_matrix
        p1 = fitz.Point(x1, y1) * inv_matrix
        p2 = fitz.Point(x2, y2) * inv_matrix

        rect = fitz.Rect(p1, p2)
        rect.normalize()
        return rect

    def process_pdf(
            self,
            pdf_path: Path,
            output_path: Optional[Path] = None,
            temp_dir: Optional[Path] = None,
    ) -> Path:
        """Process a PDF and return path to the redacted output PDF.

        The output has the same content as input, except detected regions are
        redacted (underlying text/images/line art removed if covered).

        Parameters
        ----------
        pdf_path:
            Input PDF path.
        output_path:
            Output path; if None, adds suffix '_cleaned.pdf' next to the input.
        temp_dir:
            Directory for temporary page renders; if None, a unique directory is created.

        Returns
        -------
        Path
            The output PDF path.
        """
        self.logger.info("Processing PDF: %s", pdf_path)

        doc = fitz.open(pdf_path)

        if output_path is None:
            output_path = pdf_path.parent / f"{pdf_path.stem}_cleaned.pdf"

        created_temp_dir = False
        if temp_dir is None:
            temp_dir = pdf_path.parent / f"temp_renders_{uuid.uuid4().hex}"
            created_temp_dir = True

        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                self.logger.info("  Page %d/%d (rotation=%s)", page_num + 1, len(doc), page.rotation)

                # 1) Render page to image
                img, render_matrix = self.render_page_to_image(page)

                # Save temporary image for detector
                temp_img_path = temp_dir / f"page_{page_num:04d}.png"
                img.save(temp_img_path)

                # 2) Run detector
                detections, _ = self.detector.predict_from_path(temp_img_path)
                self.logger.info("    Detected regions: %d", len(detections))

                # 3) Convert bboxes to PDF coords and add redact annotations
                redactions = 0
                for det in detections:
                    bbox = det["bbox"]  # (x1, y1, x2, y2) in image coords
                    pdf_rect = self.convert_bbox_to_pdf_coords(bbox, page, render_matrix)
                    page.add_redact_annot(pdf_rect, fill=None)
                    redactions += 1

                # 4) Apply redactions if any
                if redactions > 0:
                    page.apply_redactions(
                        text=fitz.PDF_REDACT_TEXT_REMOVE,
                        images=fitz.PDF_REDACT_IMAGE_PIXELS,
                        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                    )
                    self.logger.info("    Applied redactions: %d", redactions)
                else:
                    self.logger.info("    Nothing to redact (no detections)")

                # Remove temporary image file
                temp_img_path.unlink(missing_ok=True)

            doc.save(output_path, garbage=3, deflate=True, clean=True)
            doc.close()

            self.logger.info("Cleaned PDF saved: %s", output_path)
            return output_path

        finally:
            if created_temp_dir:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning("Failed to remove temp dir %s: %s", temp_dir, e)
