"""A page neither reader can read is left without text; the document carries on."""

import io
import subprocess

from PIL import Image

from contextread.indexing import pipeline


def _png(w=1000, h=1300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_page_neither_reader_can_read_is_left_without_text(monkeypatch):
    class Broken:
        def read_layout(self, page_no, png):
            raise RuntimeError("model unavailable")

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("tesseract", 120)

    monkeypatch.setattr(pipeline, "ocr_blocks", timeout)
    assert pipeline._read_unusable_page(Broken(), 5, b"vlm", _png()) == ([], "image")
