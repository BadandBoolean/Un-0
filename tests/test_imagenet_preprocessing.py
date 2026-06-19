from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "imagenet_preprocessing",
    Path(__file__).resolve().parents[1] / "scripts" / "imagenet_preprocessing.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
preprocess_imagenet = _MOD.preprocess_imagenet


def test_preprocess_returns_64x64x3_uint8() -> None:
    """Any source image maps to a (64, 64, 3) uint8 array."""
    img = Image.new("RGB", (500, 300), (123, 45, 67))
    out = preprocess_imagenet(img)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_preprocess_center_crops_to_square() -> None:
    """A wide solid image center-crops to a uniform 64x64 of that color."""
    img = Image.new("RGB", (400, 200), (10, 20, 30))
    out = preprocess_imagenet(img)
    # Solid color survives resize+crop (allow a small bicubic ring tolerance).
    assert abs(int(out[32, 32, 0]) - 10) <= 2
    assert abs(int(out[32, 32, 1]) - 20) <= 2
    assert abs(int(out[32, 32, 2]) - 30) <= 2


def test_preprocess_handles_grayscale_via_rgb_conversion() -> None:
    """A non-RGB image is converted, not crashed on."""
    img = Image.new("L", (128, 128), 200)
    out = preprocess_imagenet(img)
    assert out.shape == (64, 64, 3)
