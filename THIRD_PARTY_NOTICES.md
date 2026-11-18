# Уведомления о сторонних компонентах (THIRD_PARTY_NOTICES)

Приложение «до/после» является обёрткой над открытым проектом
**LandmarkDiff** и использует перечисленные ниже компоненты.
Лицензии указаны по метаданным установленных пакетов и официальным
страницам проектов. Проверено 17 ноября 2026 года.

## Обязательное уведомление LandmarkDiff

LandmarkDiff распространяется по лицензии MIT. Файл `LICENSE` в корне
репозитория сохранён без изменений:

> MIT License
> Copyright (c) 2026 dreamlessx

Исходный код: <https://github.com/dreamlessx/LandmarkDiff-public>

## Используемые компоненты

| Компонент | Версия | Лицензия | Источник |
|---|---|---|---|
| LandmarkDiff | 0.2.3 (commit `e512b1c`) | MIT | github.com/dreamlessx/LandmarkDiff-public |
| MediaPipe (Python) | 1.2.3 | Apache-2.0 | github.com/google-ai-edge/mediapipe |
| MediaPipe Face Landmarker (`models/face_landmarker.task`) | float16/1 | Apache-2.0 (модель MediaPipe) | storage.googleapis.com/mediapipe-models |
| PyTorch | 2.15.0 (CPU) | BSD-3-Clause | pytorch.org |
| OpenCV (`opencv-python`, `opencv-contrib-python`) | 5.1.0.95 | Apache-2.0 | opencv.org |
| NumPy | 2.4.6 | BSD-3-Clause | numpy.org |
| Pillow | 13.0.0 | MIT-CMU (HPND) | python-pillow.org |
| Gradio | 6.32.0 | Apache-2.0 | gradio.app |
| FastAPI | 0.163.1 | MIT | fastapi.tiangolo.com |
| Uvicorn | 0.54.0 | BSD-3-Clause | uvicorn.dev |
| Diffusers | 0.43.0 | Apache-2.0 | huggingface.co/docs/diffusers |
| Transformers | 5.23.0 | Apache-2.0 | huggingface.co/docs/transformers |
| Accelerate | 1.16.0 | Apache-2.0 | huggingface.co/docs/accelerate |
| Safetensors | 0.8.0 | Apache-2.0 | github.com/huggingface/safetensors |
| Hugging Face Hub | (зависимость Gradio/Diffusers) | Apache-2.0 | huggingface.co |
| Tokenizers | 0.23.3 | Apache-2.0 | github.com/huggingface/tokenizers |
| PyYAML | 6.0.3 | MIT | pyyaml.org |
| pytest (только тесты) | 9.2.0 | MIT | pytest.org |

`diffusers`, `transformers`, `accelerate` и `safetensors` установлены как
зависимости пакета LandmarkDiff, но в базовом TPS-режиме не используются
для загрузки весов.

## Компоненты, которые НЕ используются и не загружаются

| Компонент | Лицензия / риск | Почему отключён |
|---|---|---|
| Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`) | CreativeML OpenRAIL-M (ограничения на использование) | Требует GPU и загрузки ~4 ГБ весов; вне базового TPS-сценария |
| ControlNet (`CrucibleAI/ControlNetMediaPipeFace`) | Apache-2.0 (веса — отдельные условия) | Требует GPU и ~1.4 ГБ весов |
| ArcFace / InsightFace | Код MIT, **предобученные модели — только non-commercial** | Лицензия весов не позволяет коммерческое использование; проверка идентичности не входит в базовую задачу |
| CodeFormer | NTU S-Lab License 1.0 (non-commercial) | Некоммерческая лицензия; постобработка не требуется |
| GFPGAN | Apache-2.0 (зависимости с иными условиями) | Не входит в базовую задачу, лицензионная проверка не завершена |
| Real-ESRGAN | BSD-3-Clause (зависимости) | Апскейлинг не входит в базовую задачу |
| xformers, triton | BSD-3-Clause / MIT | Нужны только для GPU-режимов |

Ни один из перечисленных выше отключённых компонентов не скачивается и
не импортируется при работе приложения (проверяется тестом
`test_base_mode_does_not_load_arcface_or_neural_weights`).

## Данные и приватность

- Изображения обрабатываются локально и не отправляются во внешние API.
- Аналитика и телеметрия Gradio отключены (`GRADIO_ANALYTICS_ENABLED=False`).
- Единственный сетевой запрос — однократное скачивание модели MediaPipe с
  официального домена Google, если файла нет в `models/`.

## Ограничение ответственности

Приложение — исследовательский инструмент визуализации, а не медицинское
изделие. Оно не выполняет диагностику, не подбирает дозу и не заменяет
консультацию специалиста.
