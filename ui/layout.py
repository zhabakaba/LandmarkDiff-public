"""Сборка русскоязычного интерфейса «до/после» на Gradio."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from services import export, images, inference
from services.images import UserFacingError
from ui import texts

logger = logging.getLogger(__name__)


@dataclass
class _State:
    """Состояние сессии: изображения и пути к временным файлам."""

    session_dir: Path
    before: object  # PIL.Image
    after: object  # PIL.Image
    result_path: str
    comparison_path: str


def _clear_outputs(count: int) -> tuple:
    return tuple(None for _ in range(count))


def _defaults() -> tuple[str, float]:
    available = texts.ordered_procedures(inference.available_procedures())
    return texts.procedure_label(available[0]), 50.0


def on_upload(path: str | None):
    """Проверить загруженный файл и показать его предпросмотр."""
    defaults = _defaults()
    if not path:
        return (None, texts.STATUS_EMPTY, gr.update(interactive=False)) + _clear_outputs(8)

    try:
        image = images.load_upload(path)
    except UserFacingError as exc:
        return (
            None,
            f"**Ошибка.** {exc.message}",
            gr.update(interactive=False),
            *_clear_outputs(7),
            None,
        )

    preview = image.copy()
    preview.thumbnail((900, 900))
    status = (
        f"{texts.STATUS_READY}\n\nРазмер: {image.width}×{image.height} px. "
        f"Процедура по умолчанию: «{defaults[0]}», интенсивность {defaults[1]:.0f}%."
    )
    # Предпросмотр уже с учётом EXIF-ориентации.
    return (
        preview,
        status,
        gr.update(interactive=True),
        *_clear_outputs(7),
        None,
    )


def on_clear_upload():
    """Срабатывает при удалении изображения средствами компонента загрузки."""
    return (None, texts.STATUS_EMPTY, gr.update(interactive=False)) + _clear_outputs(8)


def on_generate(
    image_src,
    procedure_label: str,
    intensity: float,
    compare_mode: str,
    split_position: float,
    state: _State | None,
):
    """Выполнить TPS-преобразование и обновить все панели."""
    available = texts.ordered_procedures(inference.available_procedures())
    procedure = texts.label_to_procedure(procedure_label, available)
    split_visible = gr.update(visible=compare_mode == texts.COMPARE_SLIDER)

    if image_src is None:
        return (
            None,
            split_visible,
            *_clear_outputs(4),
            "**Нужна фотография.** Загрузите изображение JPG, PNG или WebP до 10 МБ.",
            gr.update(interactive=False),
            gr.update(interactive=False),
            state,
        )

    try:
        image = images.load_upload(image_src)
        prepared = images.prepare(image)
        result = inference.run_tps(
            prepared.bgr,
            procedure,
            intensity,
            allow_download=inference.downloads_allowed(),
        )
    except UserFacingError as exc:
        logger.info("Отказ обработки: %s", type(exc).__name__)
        return (
            None,
            split_visible,
            *_clear_outputs(4),
            f"**Не удалось обработать.** {exc.message}",
            gr.update(interactive=False),
            gr.update(interactive=False),
            state,
        )
    except MemoryError:
        logger.error("Недостаточно памяти при обработке")
        return (
            None,
            split_visible,
            *_clear_outputs(4),
            "**Недостаточно памяти.** Попробуйте фотографию меньшего размера "
            "и закройте другие приложения.",
            gr.update(interactive=False),
            gr.update(interactive=False),
            state,
        )
    except Exception:  # noqa: BLE001 - traceback только в журнал
        logger.exception("Непредвиденная ошибка обработки")
        return (
            None,
            split_visible,
            *_clear_outputs(4),
            "**Ошибка обработки.** Подробности записаны в технический журнал.",
            gr.update(interactive=False),
            gr.update(interactive=False),
            state,
        )

    session_dir = state.session_dir if state else images.create_session_dir()
    result_path = export.save_result(result.after, session_dir)
    comparison = (
        export.split_composite(result.before, result.after, split_position)
        if compare_mode == texts.COMPARE_SLIDER
        else export.side_by_side(result.before, result.after)
    )
    comparison_path = export.save_png(comparison, Path(session_dir) / "comparison.png")

    new_state = _State(
        session_dir=Path(session_dir),
        before=result.before,
        after=result.after,
        result_path=str(result_path),
        comparison_path=str(comparison_path),
    )

    status = (
        f"**Результат готов.** Процедура: «{texts.procedure_label(procedure)}», "
        f"интенсивность: {result.intensity:.0f}%. "
        f"Средний сдвиг ориентиров: {result.mean_shift_px:.2f} px, "
        f"максимальный: {result.max_shift_px:.1f} px. "
        f"Время: {result.elapsed_s:.2f} с.\n\n"
        "Это геометрическое преобразование изображения, а не медицинский прогноз."
    )

    return (
        comparison,
        split_visible,
        result.landmarks_before,
        result.landmarks_after,
        result.mask,
        result.technical,
        status,
        gr.update(value=str(result_path), interactive=True, visible=True),
        gr.update(value=str(comparison_path), interactive=True, visible=True),
        new_state,
    )


def on_compare_change(compare_mode: str, split_position: float, state: _State | None):
    """Перестроить сравнение при смене режима или положения разделителя."""
    split_visible = gr.update(visible=compare_mode == texts.COMPARE_SLIDER)
    if state is None:
        return None, gr.update(interactive=False), split_visible

    comparison = (
        export.split_composite(state.before, state.after, split_position)
        if compare_mode == texts.COMPARE_SLIDER
        else export.side_by_side(state.before, state.after)
    )
    path = export.save_png(comparison, Path(state.session_dir) / "comparison.png")
    state.comparison_path = str(path)
    return (
        comparison,
        gr.update(value=str(path), interactive=True, visible=True),
        split_visible,
    )


def on_reset(state: _State | None):
    """Сбросить результат и параметры, сохранив загруженную фотографию."""
    procedure, intensity = _defaults()
    return (
        procedure,
        intensity,
        *_clear_outputs(5),
        "Параметры сброшены. Нажмите «Сформировать результат».",
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
        state,
    )


def on_delete(state: _State | None):
    """Удалить фотографию, результаты и временные файлы сессии."""
    if state is not None:
        images.cleanup_session(state.session_dir)
    return (
        None,
        *_clear_outputs(5),
        texts.STATUS_DELETED,
        gr.update(interactive=False),
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
        None,
    )


def build_demo() -> gr.Blocks:
    """Собрать интерфейс приложения."""
    available = texts.ordered_procedures(inference.available_procedures())
    choices = [texts.procedure_label(name) for name in available]
    default_procedure, default_intensity = _defaults()

    with gr.Blocks(title=texts.APP_TITLE) as demo:
        gr.Markdown(f"# {texts.APP_TITLE}\n\n{texts.APP_INTRO}")
        gr.Markdown(texts.WARNING)
        gr.Markdown(texts.MODE_NOTE)

        state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.Image(
                    label="Фотография (JPG, PNG, WebP, до 10 МБ)",
                    type="filepath",
                    sources=["upload"],
                    height=320,
                )
                status = gr.Markdown(texts.STATUS_EMPTY)

                procedure = gr.Radio(
                    choices=choices,
                    value=default_procedure,
                    label="Процедура",
                )
                intensity = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=default_intensity,
                    step=1,
                    label="Интенсивность, %",
                )
                generate = gr.Button(
                    "Сформировать результат",
                    variant="primary",
                    interactive=False,
                )
                with gr.Row():
                    reset = gr.Button("Сбросить параметры")
                    delete = gr.Button("Удалить фотографию", variant="secondary")

            with gr.Column(scale=1):
                compare_mode = gr.Radio(
                    choices=[texts.COMPARE_SLIDER, texts.COMPARE_SIDE],
                    value=texts.COMPARE_SLIDER,
                    label="Режим сравнения",
                )
                split_position = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Положение разделителя «до/после», %",
                )
                comparison = gr.Image(
                    label="Сравнение: слева «до», справа «после»",
                    interactive=False,
                    height=420,
                )
                with gr.Row():
                    download_result = gr.DownloadButton(
                        "Скачать результат (PNG)",
                        interactive=False,
                        variant="primary",
                    )
                    download_compare = gr.DownloadButton(
                        "Скачать сравнение (PNG)",
                        interactive=False,
                    )

        with gr.Accordion("Отладка: ориентиры, маска, параметры", open=False):
            gr.Markdown(texts.DEBUG_NOTE)
            with gr.Row():
                landmarks_before = gr.Image(
                    label="Исходные ориентиры", interactive=False, height=300
                )
                landmarks_after = gr.Image(
                    label="Изменённые ориентиры", interactive=False, height=300
                )
                mask_image = gr.Image(label="Маска области", interactive=False, height=300)
            params_json = gr.JSON(label="Технические параметры запуска")

        with gr.Accordion("О программе и лицензиях", open=False):
            gr.Markdown(texts.ABOUT_TEXT)

        upload_outputs = [
            upload,
            status,
            generate,
            comparison,
            landmarks_before,
            landmarks_after,
            mask_image,
            params_json,
            download_result,
            download_compare,
            state,
        ]
        generate_outputs = [
            comparison,
            split_position,
            landmarks_before,
            landmarks_after,
            mask_image,
            params_json,
            status,
            download_result,
            download_compare,
            state,
        ]

        upload.upload(fn=on_upload, inputs=[upload], outputs=upload_outputs)
        upload.clear(fn=on_clear_upload, inputs=None, outputs=upload_outputs)

        generate.click(
            fn=on_generate,
            inputs=[upload, procedure, intensity, compare_mode, split_position, state],
            outputs=generate_outputs,
        )

        for trigger in (compare_mode.change, split_position.release):
            trigger(
                fn=on_compare_change,
                inputs=[compare_mode, split_position, state],
                outputs=[comparison, download_compare, split_position],
            )

        reset.click(
            fn=on_reset,
            inputs=[state],
            outputs=[
                procedure,
                intensity,
                comparison,
                landmarks_before,
                landmarks_after,
                mask_image,
                params_json,
                status,
                download_result,
                download_compare,
                state,
            ],
        )

        delete.click(
            fn=on_delete,
            inputs=[state],
            outputs=[
                upload,
                comparison,
                landmarks_before,
                landmarks_after,
                mask_image,
                params_json,
                status,
                generate,
                download_result,
                download_compare,
                state,
            ],
        )

        def _shutdown() -> None:
            images.cleanup_all_sessions()

        if hasattr(demo, "unload"):
            demo.unload(_shutdown)

    return demo
