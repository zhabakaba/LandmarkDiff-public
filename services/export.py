"""Экспорт результата и совмещённого сравнения «до/после»."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

DIVIDER_COLOR = (255, 255, 255)
DIVIDER_WIDTH = 2
SIDE_BY_SIDE_GAP = 8


def _same_size(before: Image.Image, after: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Привести оба изображения к одному размеру и кадрированию."""
    if after.size != before.size:
        after = after.resize(before.size, Image.LANCZOS)
    return before.convert("RGB"), after.convert("RGB")


def split_composite(before: Image.Image, after: Image.Image, position: float = 50.0) -> Image:
    """Совмещённое сравнение: слева «до», справа «после» по позиции разделителя.

    Args:
        before: исходное изображение.
        after: результат.
        position: положение разделителя в процентах (0 — только «после»,
            100 — только «до»).
    """
    before, after = _same_size(before, after)
    width, height = before.size
    cut = int(round(width * max(0.0, min(100.0, float(position))) / 100.0))
    cut = max(1, min(width - 1, cut))

    canvas = after.copy()
    canvas.paste(before.crop((0, 0, cut, height)), (0, 0))
    if DIVIDER_WIDTH:
        left = max(0, cut - DIVIDER_WIDTH // 2)
        right = min(width, left + DIVIDER_WIDTH)
        canvas.paste(Image.new("RGB", (right - left, height), DIVIDER_COLOR), (left, 0))
    return canvas


def side_by_side(before: Image.Image, after: Image.Image) -> Image:
    """Два изображения рядом: «до» слева, «после» справа."""
    before, after = _same_size(before, after)
    width, height = before.size
    canvas = Image.new(
        "RGB", (width * 2 + SIDE_BY_SIDE_GAP, height), DIVIDER_COLOR
    )
    canvas.paste(before, (0, 0))
    canvas.paste(after, (width + SIDE_BY_SIDE_GAP, 0))
    return canvas


def save_png(image: Image.Image, path: str | Path) -> Path:
    """Сохранить изображение в PNG и вернуть путь."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(target, format="PNG")
    return target


def save_result(image: Image.Image, session_dir: str | Path, name: str = "result.png") -> Path:
    """Сохранить результат процедуры в директорию сессии."""
    return save_png(image, Path(session_dir) / name)


def save_comparison(
    before: Image.Image,
    after: Image.Image,
    session_dir: str | Path,
    *,
    position: float | None = None,
    name: str = "comparison.png",
) -> Path:
    """Сохранить совмещённое сравнение (по разделителю или рядом)."""
    composite = (
        side_by_side(before, after)
        if position is None
        else split_composite(before, after, position)
    )
    return save_png(composite, Path(session_dir) / name)
