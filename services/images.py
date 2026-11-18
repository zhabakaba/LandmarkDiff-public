"""Проверка, загрузка и подготовка изображений для интерфейса «до/после».

Все проверки выполняются локально: изображение не покидает машину.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 МБ
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_SIDE = 512  # Рабочее разрешение TPS-режима

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_ROOT = PROJECT_ROOT / ".sessions"


class UserFacingError(Exception):
    """Ошибка с текстом, который можно безопасно показать пользователю."""

    default_message = "Не удалось обработать изображение."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message

    def __str__(self) -> str:  # pragma: no cover - тривиально
        return self.message


def load_upload(source: str | Path | Image.Image) -> Image.Image:
    """Загрузить загруженный файл и вернуть RGB-изображение с учётом EXIF.

    Raises:
        UserFacingError: если файл слишком большой, не поддерживается
            или повреждён.
    """
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")

    path = Path(source)
    if not path.exists():
        raise UserFacingError("Файл не найден. Загрузите изображение заново.")

    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise UserFacingError(
            "Неподдерживаемый формат файла. Допустимы JPG, PNG и WebP."
        )

    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise UserFacingError(
            f"Файл слишком большой ({size / 1024 / 1024:.1f} МБ). "
            f"Максимальный размер — {limit} МБ."
        )
    if size == 0:
        raise UserFacingError("Файл пустой. Загрузите другое изображение.")

    try:
        with Image.open(path) as raw:
            if raw.format not in ALLOWED_FORMATS:
                raise UserFacingError(
                    "Неподдерживаемый формат изображения. Допустимы JPG, PNG и WebP."
                )
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except UserFacingError:
        raise
    except Exception as exc:  # noqa: BLE001 - нужен один понятный текст
        logger.warning("Не удалось прочитать изображение: %s", type(exc).__name__)
        raise UserFacingError(
            "Не удалось прочитать изображение. Возможно, файл повреждён."
        ) from exc

    if image.width < 64 or image.height < 64:
        raise UserFacingError(
            "Изображение слишком маленькое. Минимальный размер — 64×64 пикселя."
        )
    return image


@dataclass(frozen=True)
class PreparedImage:
    """Изображение, приведённое к рабочему разрешению."""

    image: Image.Image  # RGB, уменьшено до MAX_SIDE
    bgr: np.ndarray  # (H, W, 3), uint8 — формат OpenCV/LandmarkDiff

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


def prepare(image: Image.Image, max_side: int = MAX_SIDE) -> PreparedImage:
    """Уменьшить изображение до рабочего разрешения, сохранив пропорции.

    Пропорции сохраняются: «до» и «после» показываются в одном кадре,
    поэтому обрезка не применяется.
    """
    rgb = image.convert("RGB")
    if max(rgb.size) > max_side:
        rgb = rgb.copy()
        rgb.thumbnail((max_side, max_side), Image.LANCZOS)
    bgr = np.ascontiguousarray(np.array(rgb)[:, :, ::-1])
    return PreparedImage(image=rgb, bgr=bgr)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """BGR uint8 (OpenCV) -> PIL RGB."""
    return Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """PIL RGB -> BGR uint8 (OpenCV)."""
    return np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])


def create_session_dir() -> Path:
    """Создать отдельную временную директорию сессии внутри проекта."""
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    session = SESSION_ROOT / uuid.uuid4().hex[:12]
    session.mkdir(parents=True, exist_ok=True)
    return session


def cleanup_session(session_dir: str | Path | None) -> None:
    """Удалить временные файлы сессии."""
    if not session_dir:
        return
    path = Path(session_dir)
    try:
        if path.is_dir() and path.parent == SESSION_ROOT:
            shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: BLE001 - удаление не должно ломать UI
        logger.warning("Не удалось удалить временную директорию сессии")


def cleanup_all_sessions() -> None:
    """Очистить временные директории, оставшиеся от прошлых запусков."""
    if not SESSION_ROOT.is_dir():
        return
    for child in SESSION_ROOT.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
