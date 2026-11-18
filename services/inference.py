"""Безопасная обёртка над фактическим API LandmarkDiff (TPS, CPU).

Вызываются только публичные функции библиотеки:
    extract_landmarks -> apply_procedure_preset -> warp_image_tps

Нейросетевые режимы (img2img, controlnet, controlnet_ip), ArcFace и
диффузионные веса здесь не используются и не загружаются.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from landmarkdiff.landmarks import FaceLandmarks, extract_landmarks, visualize_landmarks
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS
from landmarkdiff.masking import generate_surgical_mask

from services import deformation
from services.images import UserFacingError, bgr_to_pil

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "face_landmarker.task"
# Официальный источник модели MediaPipe Face Landmarker.
OFFICIAL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
# Именно этот путь ожидает landmarkdiff.landmarks._extract_tasks_api.
LIBRARY_TEMP_MODEL_NAME = "face_landmarker_v2_with_blendshapes.task"
# Сколько лиц проверяем, чтобы отличить «одно лицо» от «несколько лиц».
FACES_TO_COUNT = 4
# Ориентиры с весом маски ниже этого считаются «вне области операции».
_OUTSIDE_MASK_WEIGHT = 0.02


class ModelUnavailableError(UserFacingError):
    """Модель распознавания лиц недоступна локально."""

    default_message = (
        "Модель анализа лица недоступна. Проверьте папку models/ и повторите запуск."
    )


class NoFaceError(UserFacingError):
    """На фотографии не найдено лицо."""

    default_message = (
        "Лицо не найдено. Загрузите более чёткую фронтальную фотографию "
        "при хорошем освещении."
    )


class MultipleFacesError(UserFacingError):
    """На фотографии больше одного лица."""

    default_message = (
        "На фотографии найдено несколько лиц. Для корректного результата "
        "загрузите фотографию одного человека."
    )


@dataclass
class TPSResult:
    """Результат TPS-режима: изображения и технические данные."""

    before: Image.Image
    after: Image.Image
    procedure: str
    intensity: float
    landmarks_before: Image.Image | None = None
    landmarks_after: Image.Image | None = None
    mask: Image.Image | None = None
    mean_shift_px: float = 0.0
    max_shift_px: float = 0.0
    elapsed_s: float = 0.0
    technical: dict[str, object] = field(default_factory=dict)


def available_procedures() -> list[str]:
    """Реальные пресеты процедур из установленной версии LandmarkDiff."""
    return sorted(PROCEDURE_LANDMARKS)


def ensure_face_model(*, allow_download: bool = True) -> Path:
    """Гарантировать локальную модель MediaPipe и вернуть её путь.

    Модель хранится в ``models/`` внутри проекта. Копия во временной
    директории нужна потому, что библиотека ищет файл именно там.
    После первого запуска сеть не требуется.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILENAME

    if not model_path.exists():
        if not allow_download:
            raise ModelUnavailableError(
                "Файл models/face_landmarker.task отсутствует, "
                "а скачивание запрещено переменной окружения."
            )
        logger.info("Скачивание модели MediaPipe из официального источника")
        tmp_path = model_path.with_suffix(".download")
        try:
            urllib.request.urlretrieve(OFFICIAL_MODEL_URL, tmp_path)
            tmp_path.replace(model_path)
        except Exception as exc:  # noqa: BLE001
            tmp_path.unlink(missing_ok=True)
            logger.error("Не удалось скачать модель: %s", type(exc).__name__)
            raise ModelUnavailableError(
                "Не удалось скачать модель анализа лица. Проверьте соединение "
                "и повторите запуск."
            ) from exc

    library_path = Path(tempfile.gettempdir()) / LIBRARY_TEMP_MODEL_NAME
    if not library_path.exists():
        try:
            shutil.copy(model_path, library_path)
        except OSError:  # noqa: PERF203 - не критично, библиотека скачает сама
            logger.warning("Не удалось подготовить локальную копию модели")
    return model_path


def count_faces(bgr: np.ndarray, model_path: Path) -> int:
    """Посчитать лица на изображении через MediaPipe Tasks API.

    MediaPipe 1.x не содержит ``mp.solutions``, поэтому библиотечный
    ``extract_all_landmarks`` в этом окружении недоступен.
    """
    import mediapipe as mp

    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=FACES_TO_COUNT,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка детектора лиц: %s", type(exc).__name__)
        raise ModelUnavailableError(
            "Не удалось запустить анализ лица. Подробности записаны в журнал."
        ) from exc
    return len(result.face_landmarks or [])


def detect_landmarks(bgr: np.ndarray) -> FaceLandmarks:
    """Извлечь 478 ориентиров через API LandmarkDiff."""
    try:
        face = extract_landmarks(bgr)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка извлечения ориентиров: %s", type(exc).__name__)
        raise ModelUnavailableError(
            "Не удалось выполнить анализ лица. Подробности записаны в журнал."
        ) from exc
    if face is None:
        raise NoFaceError()
    return face


def _landmark_overlay(bgr: np.ndarray, face: FaceLandmarks) -> Image.Image:
    return bgr_to_pil(visualize_landmarks(bgr, face, radius=1))


def _mask_overlay(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    try:
        mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(mask_uint8, mode="L").convert("RGB")
    except Exception as exc:  # noqa: BLE001 - маска нужна только для отладки
        logger.warning("Маска недоступна: %s", type(exc).__name__)
        return Image.new("RGB", size, (0, 0, 0))


def run_tps(
    bgr: np.ndarray,
    procedure: str,
    intensity: float,
    *,
    allow_download: bool = True,
) -> TPSResult:
    """Выполнить TPS-преобразование на CPU.

    Интенсивность 0 возвращает исходное изображение без обработки.
    """
    if procedure not in PROCEDURE_LANDMARKS:
        raise UserFacingError("Выбранная процедура недоступна в этой версии.")

    intensity = float(np.clip(float(intensity), 0.0, 100.0))
    height, width = bgr.shape[:2]
    before_image = bgr_to_pil(bgr)

    if intensity == 0.0:
        logger.info("Интенсивность 0: изображение возвращено без изменений")
        return TPSResult(
            before=before_image,
            after=before_image.copy(),
            procedure=procedure,
            intensity=0.0,
            technical={
                "mode": "tps",
                "device": "cpu",
                "image_size": [width, height],
                "intensity": 0.0,
                "note": "Изображение не изменялось (интенсивность 0).",
            },
        )

    model_path = ensure_face_model(allow_download=allow_download)

    faces = count_faces(bgr, model_path)
    if faces == 0:
        raise NoFaceError()
    if faces > 1:
        raise MultipleFacesError()

    started = time.monotonic()
    face = detect_landmarks(bgr)
    mask = generate_surgical_mask(face, procedure, width, height)
    manipulated = deformation.apply_procedure(
        face, procedure, intensity, image_size=max(width, height)
    )
    warped, weights = deformation.localized_warp(bgr, face, manipulated, mask)
    elapsed = time.monotonic() - started

    shift = np.linalg.norm(manipulated.pixel_coords - face.pixel_coords, axis=1) * weights
    outside = shift[weights < _OUTSIDE_MASK_WEIGHT]
    inside = shift[weights >= 0.5]
    technical: dict[str, object] = {
        "mode": "tps",
        "device": "cpu",
        "image_size": [width, height],
        "procedure": procedure,
        "intensity": intensity,
        "landmarks": int(face.landmarks.shape[0]),
        "confidence": round(float(face.confidence), 3),
        "localized": True,
        "mean_shift_px": round(float(shift.mean()), 3),
        "max_shift_px": round(float(shift.max()), 3),
        "inside_shift_px": round(float(inside.mean()) if inside.size else 0.0, 3),
        "outside_shift_px": round(float(outside.mean()) if outside.size else 0.0, 3),
        "mask_fraction": round(float((mask > 0.01).mean()), 3),
        "elapsed_s": round(elapsed, 2),
        "library_version": _library_version(),
    }

    return TPSResult(
        before=before_image,
        after=bgr_to_pil(warped),
        procedure=procedure,
        intensity=intensity,
        landmarks_before=_landmark_overlay(bgr, face),
        landmarks_after=_landmark_overlay(bgr, manipulated),
        mask=_mask_overlay(mask, (width, height)),
        mean_shift_px=float(shift.mean()),
        max_shift_px=float(shift.max()),
        elapsed_s=elapsed,
        technical=technical,
    )


def _library_version() -> str:
    try:
        import landmarkdiff

        return str(getattr(landmarkdiff, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def downloads_allowed() -> bool:
    """Скачивание модели можно запретить переменной окружения."""
    return os.environ.get("LANDMARKDIFF_NO_DOWNLOAD", "").strip() not in {"1", "true", "yes"}
