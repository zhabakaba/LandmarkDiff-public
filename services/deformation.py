"""Локальная анатомическая деформация поверх библиотечного TPS-режима.

Модуль исправляет две причины, из-за которых результат выглядел как
«увеличение/уменьшение картинки», а не как изменение после процедуры:

* ``apply_procedure`` — для приоритетных процедур (подбородок, веки, брови,
  ортогнатика) смещения ориентиров задаются здесь. Библиотечные пресеты ``mentoplasty`` и
  ``genioplasty`` двигали подбородок вверх (−Y) и тем самым укорачивали нижнюю
  треть лица, то есть буквально уменьшали кадр.
* ``localized_warp`` — поле TPS ограничивается областью операции: смещения
  ориентиров вне феатерованной маски обнуляются, а результат смешивается с
  исходником по маске. Поэтому ткани вокруг операции не растягиваются.

Смещения перекрывающихся RBF-хэндлов усредняются, а не суммируются: иначе
соседние хэндлы складывались и давали сдвиг в 2–3 раза больше заданного.

Код библиотеки LandmarkDiff не изменяется — всё реализовано обёрткой.
"""

from __future__ import annotations

import math

import numpy as np
from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.synthetic.tps_warp import warp_image_tps

REFERENCE_SIZE = 512.0

# Процедуры, для которых направление сдвигов задаётся в этом модуле.
PRIORITY_PROCEDURES = frozenset(
    {
        "mentoplasty",
        "genioplasty",
        "blepharoplasty",
        "brow_lift",
        "orthognathic",
        "rhytidectomy",
    }
)

# (индекс ориентира, dx, dy, множитель радиуса влияния);
# dx/dy — пиксели при опорном размере 512 и интенсивности 100.
_PRESETS: dict[str, tuple[tuple[int, float, float, float], ...]] = {
    # Имплантат подбородка: нижняя треть удлиняется, контур подбородка
    # опускается вслед за вершиной.
    "mentoplasty": (
        (152, 0.0, 8.0, 1.0),
        (175, 0.0, 8.0, 1.0),
        (148, 0.0, 5.5, 0.8),
        (149, 0.0, 5.5, 0.8),
        (150, 0.0, 5.5, 0.8),
        (176, 0.0, 5.5, 0.8),
        (377, 0.0, 5.5, 0.8),
        (171, 0.0, 2.5, 0.6),
        (396, 0.0, 2.5, 0.6),
    ),
    # Костная гениопластика: смещение шире и затрагивает линию челюсти.
    "genioplasty": (
        (152, 0.0, 9.0, 1.0),
        (175, 0.0, 9.0, 1.0),
        (199, 0.0, 7.0, 1.0),
        (200, 0.0, 7.0, 1.0),
        (17, 0.0, 3.0, 0.7),
        (314, 0.0, 3.0, 0.7),
        (405, 0.0, 3.0, 0.7),
        (149, 0.0, 4.5, 0.6),
        (150, 0.0, 4.5, 0.6),
        (176, 0.0, 4.5, 0.6),
        (377, 0.0, 4.5, 0.6),
        (378, 0.0, 4.5, 0.6),
        (396, 0.0, 4.5, 0.6),
    ),
    # Верхнее веко поднимается, нижнее подтягивается вверх, а не вниз:
    # вниз — это уже не блефаропластика, а расширение глаза.
    "blepharoplasty": (
        (159, 0.0, -4.5, 1.0),
        (160, 0.0, -4.5, 1.0),
        (161, 0.0, -4.5, 1.0),
        (386, 0.0, -4.5, 1.0),
        (385, 0.0, -4.5, 1.0),
        (384, 0.0, -4.5, 1.0),
        (158, 0.0, -2.0, 0.7),
        (157, 0.0, -2.0, 0.7),
        (133, 0.0, -2.0, 0.7),
        (33, 0.0, -2.0, 0.7),
        (387, 0.0, -2.0, 0.7),
        (388, 0.0, -2.0, 0.7),
        (362, 0.0, -2.0, 0.7),
        (263, 0.0, -2.0, 0.7),
        (145, 0.0, -1.2, 0.5),
        (153, 0.0, -1.2, 0.5),
        (154, 0.0, -1.2, 0.5),
        (374, 0.0, -1.2, 0.5),
        (380, 0.0, -1.2, 0.5),
        (381, 0.0, -1.2, 0.5),
    ),
    # Подъём брови: латеральная часть выше медиальной, лоб почти не тянется —
    # кожу при этой операции иссекают, а не растягивают.
    "brow_lift": (
        (70, 0.0, -5.8, 1.0),
        (63, 0.0, -6.6, 1.0),
        (105, 0.0, -7.4, 1.0),
        (66, 0.0, -8.2, 1.0),
        (107, 0.0, -9.0, 1.0),
        (300, 0.0, -5.8, 1.0),
        (293, 0.0, -6.6, 1.0),
        (334, 0.0, -7.4, 1.0),
        (296, 0.0, -8.2, 1.0),
        (336, 0.0, -9.0, 1.0),
        (9, 0.0, -1.5, 1.2),
        (8, 0.0, -1.5, 1.2),
        (10, 0.0, -1.5, 1.2),
        (109, 0.0, -1.5, 1.2),
        (67, 0.0, -1.5, 1.2),
        (103, 0.0, -1.5, 1.2),
        (338, 0.0, -1.5, 1.2),
        (297, 0.0, -1.5, 1.2),
        (332, 0.0, -1.5, 1.2),
    ),
    # Подтяжка лица: библиотечный пресет суммировал 12 хэндлов с радиусом 40 px
    # и тянул вверх всю нижнюю треть. Здесь поднимаются только контур челюсти
    # у ушей и виски, а вершина подбородка остаётся на месте.
    "rhytidectomy": (
        (132, 1.5, -4.0, 1.0),
        (136, 1.2, -4.0, 1.0),
        (172, 1.0, -4.0, 1.0),
        (58, 1.0, -4.0, 1.0),
        (150, 0.8, -3.5, 1.0),
        (176, 0.8, -3.5, 1.0),
        (361, -1.5, -4.0, 1.0),
        (365, -1.2, -4.0, 1.0),
        (397, -1.0, -4.0, 1.0),
        (379, -1.0, -4.0, 1.0),
        (400, -0.8, -3.5, 1.0),
        (10, 0.0, -0.6, 0.7),
        (21, 0.0, -0.6, 0.7),
        (54, 0.0, -0.6, 0.7),
        (67, 0.0, -0.6, 0.7),
        (103, 0.0, -0.6, 0.7),
        (109, 0.0, -0.6, 0.7),
        (162, 0.0, -0.6, 0.7),
        (127, 0.0, -0.6, 0.7),
        (284, 0.0, -0.6, 0.7),
        (297, 0.0, -0.6, 0.7),
        (332, 0.0, -0.6, 0.7),
        (338, 0.0, -0.6, 0.7),
        (323, 0.0, -0.6, 0.7),
        (356, 0.0, -0.6, 0.7),
        (389, 0.0, -0.6, 0.7),
        (454, 0.0, -0.6, 0.7),
    ),
    # Ортогнатика: библиотечный пресет суммировал перекрывающиеся хэндлы
    # и уводил левую сторону челюсти вверх вдвое сильнее правой. Здесь
    # сдвиги симметричны, а усреднение не даёт им складываться.
    "orthognathic": (
        (17, 0.0, -4.0, 1.0),
        (18, 0.0, -4.0, 1.0),
        (200, 0.0, -4.0, 1.0),
        (201, 0.0, -4.0, 1.0),
        (202, 0.0, -4.0, 1.0),
        (204, 0.0, -4.0, 1.0),
        (208, 0.0, -4.0, 1.0),
        (211, 0.0, -4.0, 1.0),
        (212, 0.0, -4.0, 1.0),
        (214, 0.0, -4.0, 1.0),
        (175, 0.0, -2.5, 0.8),
        (170, 0.0, -2.5, 0.8),
        (169, 0.0, -2.5, 0.8),
        (167, 0.0, -2.5, 0.8),
        (396, 0.0, -2.5, 0.8),
        (57, 1.5, -1.5, 0.8),
        (61, 1.5, -1.5, 0.8),
        (78, 1.5, -1.5, 0.8),
        (91, 1.5, -1.5, 0.8),
        (95, 1.5, -1.5, 0.8),
        (146, 1.5, -1.5, 0.8),
        (181, 1.5, -1.5, 0.8),
        (291, -1.5, -1.5, 0.8),
        (311, -1.5, -1.5, 0.8),
        (312, -1.5, -1.5, 0.8),
        (321, -1.5, -1.5, 0.8),
        (324, -1.5, -1.5, 0.8),
        (325, -1.5, -1.5, 0.8),
        (375, -1.5, -1.5, 0.8),
        (405, -1.5, -1.5, 0.8),
    ),
}

# Базовый радиус влияния (пиксели при 512), как в библиотеке.
_BASE_RADIUS = {
    "mentoplasty": 22.0,
    "genioplasty": 26.0,
    "blepharoplasty": 13.0,
    "brow_lift": 25.0,
    "orthognathic": 30.0,
    "rhytidectomy": 20.0,
}


def apply_procedure(
    face: FaceLandmarks,
    procedure: str,
    intensity: float,
    image_size: int = 512,
) -> FaceLandmarks:
    """Сместить ориентиры под процедуру.

    Для приоритетных процедур используется локальная таблица сдвигов,
    для остальных — библиотечный пресет без изменений.
    """
    if procedure not in PRIORITY_PROCEDURES:
        return apply_procedure_preset(
            face, procedure, intensity, image_size=image_size
        )

    scale = float(np.clip(intensity, 0.0, 100.0)) / 100.0
    if scale == 0.0:
        return FaceLandmarks(
            landmarks=face.landmarks.copy(),
            image_width=face.image_width,
            image_height=face.image_height,
            confidence=face.confidence,
        )

    width, height = face.image_width, face.image_height
    geo_mean = math.sqrt(max(width * height, 1))
    pixel_scale = geo_mean / REFERENCE_SIZE

    landmarks = face.landmarks.copy()
    pixel = landmarks.copy()
    pixel[:, 0] *= width
    pixel[:, 1] *= height

    confidence = np.asarray(face.landmark_confidence, dtype=np.float64)
    handles: list[tuple[int, float, float, float]] = []
    for idx, dx, dy, radius_factor in _PRESETS[procedure]:
        anchor_confidence = float(confidence[idx])
        handles.append(
            (
                idx,
                dx * scale * pixel_scale * anchor_confidence,
                dy * scale * pixel_scale * anchor_confidence,
                _BASE_RADIUS[procedure] * radius_factor * pixel_scale,
            )
        )

    deformed = _apply_handles(pixel, handles)

    result = deformed.copy()
    result[:, 0] /= width
    result[:, 1] /= height
    return FaceLandmarks(
        landmarks=result,
        image_width=width,
        image_height=height,
        confidence=face.confidence,
    )


def _apply_handles(
    pixel_landmarks: np.ndarray,
    handles: list[tuple[int, float, float, float]],
) -> np.ndarray:
    """Нормированная RBF-деформация: перекрытия усредняются, а не суммируются.

    Суммирование, как в библиотеке, давало сдвиг в 2–3 раза больше заданного
    там, где хэндлы перекрываются (подбородок, веки).
    """
    if not handles:
        return pixel_landmarks.copy()

    centers = np.array(
        [pixel_landmarks[idx, :2] for idx, *_ in handles], dtype=np.float64
    )
    displacements = np.array(
        [[dx, dy] for _, dx, dy, _ in handles], dtype=np.float64
    )
    radii = np.array([radius for *_, radius in handles], dtype=np.float64)

    diff = pixel_landmarks[np.newaxis, :, :2] - centers[:, np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=2)
    weights = np.exp(-dist_sq / (2.0 * np.maximum(radii, 1e-3)[:, np.newaxis] ** 2))

    weighted = np.einsum("nh,nc->hc", weights, displacements)
    denominator = np.maximum(weights.sum(axis=0)[:, np.newaxis], 1.0)
    delta = weighted / denominator

    result = pixel_landmarks.copy()
    result[:, :2] += delta
    return result


def localized_warp(
    bgr: np.ndarray,
    face: FaceLandmarks,
    manipulated: FaceLandmarks,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Деформировать кадр только внутри области операции.

    Возвращает изображение и вес каждого ориентира (0 — не двигался,
    1 — сдвинут полностью). Вне маски TPS-поле единичное, поэтому фон и
    соседние области не растягиваются.
    """
    src = face.pixel_coords
    dst = manipulated.pixel_coords
    height, width = bgr.shape[:2]

    ys = np.clip(np.rint(src[:, 1]).astype(int), 0, height - 1)
    xs = np.clip(np.rint(src[:, 0]).astype(int), 0, width - 1)
    # Запас 1.5 нужен, потому что феатерованная маска на самих ориентирах
    # уже слегка размыта и даёт значения меньше единицы.
    weights = np.clip(mask[ys, xs].astype(np.float64) * 1.5, 0.0, 1.0)

    constrained = src + weights[:, np.newaxis] * (dst - src)
    warped = warp_image_tps(bgr, src, constrained)

    alpha = mask[..., np.newaxis].astype(np.float32)
    original = bgr.astype(np.float32)
    blended = warped.astype(np.float32) * alpha + original * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8), weights
