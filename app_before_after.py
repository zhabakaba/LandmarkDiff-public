"""Точка входа русскоязычного интерфейса «до/после» (TPS, CPU).

Запуск:
    .venv/Scripts/python.exe app_before_after.py
Приложение доступно только на 127.0.0.1.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / ".logs"

# Никакой аналитики и телеметрии Gradio.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


def configure_logging(verbose: bool = False) -> None:
    """Технический журнал в файл: без изображений и персональных данных."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    ]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LandmarkDiff: интерфейс «до/после» (TPS, CPU)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Порт (по умолчанию 7860)")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Адрес прослушивания (по умолчанию только локальный 127.0.0.1)",
    )
    parser.add_argument("--verbose", action="store_true", help="Дублировать журнал в консоль")
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Открыть браузер после запуска",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import gradio as gr

    args = parse_args(argv)
    configure_logging(args.verbose)

    if args.host not in {"127.0.0.1", "localhost"}:
        logging.getLogger(__name__).warning(
            "Приложение доступно не только локально (%s): изображения может "
            "увидеть кто угодно в сети.",
            args.host,
        )

    from services import images, inference
    from ui.layout import build_demo

    images.cleanup_all_sessions()
    if inference.downloads_allowed():
        try:
            inference.ensure_face_model()
        except Exception as exc:  # noqa: BLE001 - сообщим в консоли
            logging.getLogger(__name__).warning("Модель не подготовлена: %s", exc)

    demo = build_demo()
    demo.queue(max_size=8, default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=args.inbrowser,
        show_error=False,
        quiet=False,
        theme=gr.themes.Soft(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
