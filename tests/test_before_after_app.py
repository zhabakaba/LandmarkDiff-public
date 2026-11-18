"""Тесты приложения «до/после» (TPS, CPU).

Используется только тестовое изображение из репозитория
(``tests/fixtures/face.png``) — фрагмент демо-панели LandmarkDiff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from services import deformation, export, images, inference
from services.images import UserFacingError
from ui import texts
from ui.layout import on_delete, on_generate, on_upload

FIXTURE = Path(__file__).parent / "fixtures" / "face.png"


@pytest.fixture(scope="module")
def prepared():
    """Подготовленное тестовое изображение (BGR, рабочее разрешение)."""
    return images.prepare(images.load_upload(FIXTURE))


def _arrays_equal(a: Image.Image, b: Image.Image) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_landmarkdiff_imports_in_project_venv():
    import landmarkdiff

    assert landmarkdiff.__version__
    assert inference.available_procedures(), "пресеты процедур должны быть доступны"


def test_all_displayed_procedures_exist_in_library():
    available = set(inference.available_procedures())
    missing = sorted(set(texts.PROCEDURE_LABELS) - available)
    assert not missing, f"В интерфейсе есть недоступные процедуры: {missing}"


def test_intensity_zero_returns_original(prepared):
    result = inference.run_tps(prepared.bgr, "rhinoplasty", 0)
    assert _arrays_equal(result.before, result.after)
    assert result.technical["mode"] == "tps"


@pytest.mark.parametrize("procedure", inference.available_procedures())
def test_every_procedure_runs_on_fixture(prepared, procedure):
    result = inference.run_tps(prepared.bgr, procedure, 40)
    assert result.after.size == result.before.size
    assert result.mean_shift_px > 0.0, f"{procedure}: ориентиры не изменились"
    assert result.technical["device"] == "cpu"


def test_intensity_affects_result(prepared):
    weak = inference.run_tps(prepared.bgr, "rhinoplasty", 20)
    strong = inference.run_tps(prepared.bgr, "rhinoplasty", 90)
    assert strong.mean_shift_px > weak.mean_shift_px


def _landmark_dy(face, moved, indices, height: int) -> np.ndarray:
    return (moved.landmarks[indices, 1] - face.landmarks[indices, 1]) * height


@pytest.mark.parametrize("procedure", ["mentoplasty", "genioplasty"])
def test_chin_procedures_lengthen_lower_face(prepared, procedure):
    """Подбородок опускается: библиотечные пресеты двигали его вверх и укорачивали лицо."""
    face = inference.detect_landmarks(prepared.bgr)
    height = prepared.bgr.shape[0]
    moved = deformation.apply_procedure(face, procedure, 70, image_size=max(prepared.size))

    dy = _landmark_dy(face, moved, [152, 175], height)
    assert np.all(dy > 0.5), f"{procedure}: подбородок должен опускаться, dy={dy.round(2)}"


def test_blepharoplasty_tightens_both_lids_upwards(prepared):
    face = inference.detect_landmarks(prepared.bgr)
    height = prepared.bgr.shape[0]
    moved = deformation.apply_procedure(face, "blepharoplasty", 70, max(prepared.size))

    upper = _landmark_dy(face, moved, [159, 160, 161, 386, 385, 384], height)
    lower = _landmark_dy(face, moved, [145, 153, 154, 374, 380, 381], height)
    assert np.all(upper < -0.3), f"верхнее веко не поднялось: {upper.round(2)}"
    assert np.all(lower < 0.0), f"нижнее веко опустилось вместо подтяжки: {lower.round(2)}"


def test_brow_lift_raises_brow_more_than_forehead(prepared):
    face = inference.detect_landmarks(prepared.bgr)
    height = prepared.bgr.shape[0]
    moved = deformation.apply_procedure(face, "brow_lift", 90, max(prepared.size))

    brow = abs(float(_landmark_dy(face, moved, [107], height)[0]))
    forehead = abs(float(_landmark_dy(face, moved, [9], height)[0]))
    assert brow > forehead, "бровь должна подниматься сильнее, чем лоб"


def test_orthognathic_moves_both_jaw_sides_symmetrically(prepared):
    """Библиотечный пресет уводил левую сторону челюсти вдвое сильнее правой."""
    face = inference.detect_landmarks(prepared.bgr)
    height, width = prepared.bgr.shape[:2]
    moved = deformation.apply_procedure(face, "orthognathic", 70, max(prepared.size))

    left = [57, 61, 78, 91, 95, 146, 181]
    right = [291, 311, 312, 321, 324, 325, 375, 405]
    left_dy = _landmark_dy(face, moved, left, height)
    right_dy = _landmark_dy(face, moved, right, height)
    left_dx = (moved.landmarks[left, 0] - face.landmarks[left, 0]) * width
    right_dx = (moved.landmarks[right, 0] - face.landmarks[right, 0]) * width

    assert np.all(left_dy < -0.5) and np.all(right_dy < -0.5), "челюсть должна подниматься"
    assert abs(left_dy.mean() - right_dy.mean()) < 1.0, (
        f"челюсть перекошена: слева {left_dy.mean():.2f}, справа {right_dy.mean():.2f}"
    )
    assert abs(left_dx.mean() + right_dx.mean()) < 1.0, "сужение челюсти должно быть симметричным"
    shift = np.linalg.norm(moved.pixel_coords - face.pixel_coords, axis=1)
    assert shift.max() < 6.0, f"ортогнатика сдвигает слишком сильно: {shift.max():.1f} px"


def test_rhytidectomy_lifts_jowl_without_squashing_mouth(prepared):
    """Подтяжка поднимает контур челюсти, но не тянет вверх рот и подбородок."""
    face = inference.detect_landmarks(prepared.bgr)
    height = prepared.bgr.shape[0]
    moved = deformation.apply_procedure(face, "rhytidectomy", 100, max(prepared.size))

    jowl = [132, 136, 172, 58, 361, 365, 397, 379]
    mouth = [61, 291, 13, 14]
    jowl_dy = _landmark_dy(face, moved, jowl, height)
    mouth_dy = _landmark_dy(face, moved, mouth, height)

    assert np.all(jowl_dy < -0.5), "контур челюсти должен подниматься"
    assert np.all(np.abs(mouth_dy) < 1.0), f"рот не должен смещаться: {mouth_dy.round(2)}"
    shift = np.linalg.norm(moved.pixel_coords - face.pixel_coords, axis=1)
    assert shift.max() < 6.0, f"подтяжка слишком сильно растягивает кадр: {shift.max():.1f} px"


def test_warp_stays_inside_surgical_mask(prepared):
    """Вне области операции пиксели не меняются — нет эффекта «зума» всего кадра."""
    result = inference.run_tps(prepared.bgr, "genioplasty", 70)
    mask = np.asarray(result.mask.convert("L"))
    outside = mask == 0
    assert outside.any()

    before = np.asarray(result.before).astype(int)
    after = np.asarray(result.after).astype(int)
    diff = np.abs(after - before).max(axis=2)
    assert diff[outside].max() <= 1, "изменились пиксели вне маски операции"
    assert result.technical["outside_shift_px"] <= 0.01
    assert result.technical["inside_shift_px"] > 0.0


def test_localized_warp_ignores_landmarks_outside_mask(prepared):
    """При нулевой маске TPS-поле вырождается в тождественное."""
    face = inference.detect_landmarks(prepared.bgr)
    height, width = prepared.bgr.shape[:2]
    moved = deformation.apply_procedure(face, "mentoplasty", 100, max(prepared.size))

    warped, weights = deformation.localized_warp(
        prepared.bgr, face, moved, np.zeros((height, width), dtype=np.float32)
    )
    assert not weights.any()
    assert _arrays_equal(images.bgr_to_pil(warped), images.bgr_to_pil(prepared.bgr))


def test_invalid_file_rejected(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(UserFacingError):
        images.load_upload(broken)


def test_unsupported_extension_rejected(tmp_path):
    text_file = tmp_path / "photo.txt"
    text_file.write_text("привет")
    with pytest.raises(UserFacingError):
        images.load_upload(text_file)


def test_oversized_file_rejected(tmp_path):
    big = tmp_path / "big.png"
    big.write_bytes(b"\x00" * (images.MAX_UPLOAD_BYTES + 1))
    with pytest.raises(UserFacingError):
        images.load_upload(big)


def test_missing_face_handled(tmp_path):
    blank = tmp_path / "blank.png"
    Image.new("RGB", (256, 256), (128, 128, 128)).save(blank)
    prepared = images.prepare(images.load_upload(blank))
    with pytest.raises(inference.NoFaceError):
        inference.run_tps(prepared.bgr, "rhinoplasty", 50)


def test_multiple_faces_rejected(tmp_path):
    face = Image.open(FIXTURE).convert("RGB")
    canvas = Image.new("RGB", (face.width * 2 + 16, face.height), (30, 30, 30))
    canvas.paste(face, (0, 0))
    canvas.paste(face, (face.width + 16, 0))
    double = tmp_path / "double.png"
    canvas.save(double)

    prepared = images.prepare(images.load_upload(double))
    with pytest.raises(inference.MultipleFacesError):
        inference.run_tps(prepared.bgr, "rhinoplasty", 50)


def test_result_saved_as_png(prepared, tmp_path):
    result = inference.run_tps(prepared.bgr, "blepharoplasty", 50)
    path = export.save_result(result.after, tmp_path)
    assert path.suffix == ".png" and path.exists()
    with Image.open(path) as saved:
        assert saved.format == "PNG"
        assert saved.size == result.after.size


def test_comparison_images_created(prepared, tmp_path):
    result = inference.run_tps(prepared.bgr, "mentoplasty", 50)
    side = export.side_by_side(result.before, result.after)
    split = export.split_composite(result.before, result.after, 40)

    assert side.width > result.before.width
    assert side.height == result.before.height
    assert split.size == result.before.size

    side_path = export.save_comparison(result.before, result.after, tmp_path)
    split_path = export.save_comparison(
        result.before, result.after, tmp_path, position=40, name="split.png"
    )
    assert side_path.exists() and split_path.exists()


def test_session_cleanup_removes_temp_files(prepared):
    session = images.create_session_dir()
    result = inference.run_tps(prepared.bgr, "rhinoplasty", 30)
    path = export.save_result(result.after, session)
    assert path.exists()

    images.cleanup_session(session)
    assert not session.exists()


def test_ui_generate_and_delete_cycle(prepared):
    out = on_generate(
        str(FIXTURE), "Ринопластика", 60, texts.COMPARE_SLIDER, 50, None
    )
    comparison, _split, _lb, _la, _mask, technical, status, dl_result, dl_compare, state = out
    assert comparison is not None
    assert technical["mode"] == "tps"
    assert "Результат готов" in status
    assert dl_result["value"].endswith("result.png")
    assert dl_compare["value"].endswith("comparison.png")

    session_dir = state.session_dir
    assert session_dir.exists()

    on_delete(state)
    assert not session_dir.exists()


def test_ui_upload_reports_size():
    preview, status, generate_btn, *_ = on_upload(str(FIXTURE))
    assert preview is not None
    assert "Фотография загружена" in status
    assert generate_btn["interactive"] is True


def test_base_mode_does_not_load_arcface_or_neural_weights(prepared):
    """Базовый сценарий не должен подключать ArcFace и нейросетевые веса."""
    inference.run_tps(prepared.bgr, "rhinoplasty", 50)

    forbidden = [
        "insightface",
        "onnxruntime",
        "landmarkdiff.arcface_torch",
        "landmarkdiff.face_verifier",
        "landmarkdiff.inference",
    ]
    loaded = [name for name in forbidden if name in sys.modules]
    assert not loaded, f"Загружены модули вне базового режима: {loaded}"
