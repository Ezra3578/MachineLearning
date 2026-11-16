"""Shared figure saving for Workshop 6.

Matplotlib writes RGBA PNGs. pdfTeX on Windows fails with
``fflush() failed`` when it embeds a large RGBA image, so every figure is
flattened to RGB before it is written to its final path.
"""

from __future__ import annotations

from pathlib import Path


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}__rgba.png")
    fig.savefig(tmp, dpi=dpi, facecolor="white")
    image = Image.open(tmp)
    image.load()
    image.convert("RGB").save(path)
    tmp.unlink()
    return path
