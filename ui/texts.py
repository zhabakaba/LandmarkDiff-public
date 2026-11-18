"""Русскоязычные тексты интерфейса «до/после»."""

from __future__ import annotations

APP_TITLE = "LandmarkDiff — визуализация «до/после»"

APP_INTRO = (
    "Загрузите фронтальную фотографию, выберите процедуру и интенсивность, "
    "затем сравните исходное изображение с результатом."
)

WARNING = (
    "> **Важно.** Визуализация создана геометрическим преобразованием "
    "изображения. Она предназначена для обсуждения со специалистом и не "
    "является медицинским прогнозом или гарантией результата."
)

MODE_NOTE = (
    "Текущий режим: **TPS, только CPU**. Это геометрическое преобразование "
    "изображения. Нейросетевые режимы (ControlNet, img2img) и проверка "
    "идентичности по ArcFace в этой сборке не используются."
)

# Русские названия фактических пресетов LandmarkDiff.
PROCEDURE_LABELS: dict[str, str] = {
    "rhinoplasty": "Ринопластика",
    "blepharoplasty": "Блефаропластика",
    "rhytidectomy": "Ритидэктомия (подтяжка лица)",
    "orthognathic": "Ортогнатическая операция",
    "brow_lift": "Поднятие бровей",
    "mentoplasty": "Ментопластика",
    "alarplasty": "Аларпластика",
    "canthoplasty": "Кантопластика",
    "buccal_fat_removal": "Удаление комков Биша",
    "dimpleplasty": "Димлпластика",
    "genioplasty": "Гениопластика",
    "malarplasty": "Маларпластика",
    "lip_lift": "Подтяжка верхней губы",
    "lip_augmentation": "Увеличение губ",
    "forehead_reduction": "Понижение линии роста волос",
    "submental_liposuction": "Липосакция подбородка",
    "otoplasty": "Отопластика",
}


def procedure_label(name: str) -> str:
    """Русская подпись процедуры (или техническое имя, если перевода нет)."""
    return PROCEDURE_LABELS.get(name, name.replace("_", " "))


def ordered_procedures(available: list[str]) -> list[str]:
    """Упорядочить фактические пресеты: основные процедуры — первыми."""
    known = [name for name in PROCEDURE_LABELS if name in available]
    extra = sorted(name for name in available if name not in PROCEDURE_LABELS)
    return known + extra


def label_to_procedure(label: str, available: list[str]) -> str:
    """Обратное преобразование подписи в имя пресета."""
    for name in available:
        if procedure_label(name) == label:
            return name
    return label if label in available else available[0]


COMPARE_SLIDER = "Слайдер «до/после»"
COMPARE_SIDE = "Два изображения рядом"

STATUS_EMPTY = "Фотография не загружена. Загрузите JPG, PNG или WebP до 10 МБ."
STATUS_READY = "Фотография загружена. Можно формировать результат."
STATUS_ANALYZING = "Анализ фотографии и генерация результата…"
STATUS_NOTHING_TO_RESET = "Нечего сбрасывать: фотография ещё не загружена."
STATUS_DELETED = "Фотография и результаты удалены. Загрузите новое изображение."
STATUS_STALE = "Параметры изменены. Нажмите «Сформировать результат», чтобы обновить."

ABOUT_TEXT = """
### О программе и лицензиях

Приложение — тонкая обёртка над открытым проектом
[LandmarkDiff](https://github.com/dreamlessx/LandmarkDiff-public) (лицензия MIT,
уведомление сохранено в файле `LICENSE`).

Используемые компоненты и лицензии перечислены в `THIRD_PARTY_NOTICES.md`.
Обработка изображений выполняется локально: ни фотография, ни результат не
отправляются во внешние сервисы, аналитика и телеметрия отключены.

Отключено в этой сборке и почему:

- **ControlNet / img2img** — требуют GPU и загрузки весов Stable Diffusion;
  базовая задача ограничена TPS-режимом.
- **ArcFace / InsightFace** — лицензия загружаемых весов ограничивает
  коммерческое использование; проверка идентичности не входит в базовый сценарий.
- **CodeFormer / GFPGAN / Real-ESRGAN** — постобработка не используется,
  лицензии не проверялись для коммерческой сборки.

Приложение не является медицинским изделием. Оно не выполняет диагностику,
не подбирает дозу и не назначает лечение.
"""

DEBUG_NOTE = (
    "Технический режим для проверки результата: исходные и изменённые "
    "ориентиры, маска и параметры запуска."
)
