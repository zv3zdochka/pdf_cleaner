"""Compatibility shim.

The original project imported RFDetrONNX as::

    from rfdetr import RFDetrONNX

To avoid breaking external imports, we keep this module and re-export
the class from its new location.
"""

from pdf_cleaner_bot.ml.rfdetr_onnx import RFDetrONNX

__all__ = ["RFDetrONNX"]
