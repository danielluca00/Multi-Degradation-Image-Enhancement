# generate_paired_degradation_dataset.py
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
CLEAN_DIR = Path("clean_images")
OUTPUT_ROOT = Path("paired_datasets")

# target final size (H, W)
TARGET_SIZE = (256, 384)

# train/test split
TEST_RATIO = 0.15
SEED = 42

# output image format
OUTPUT_EXT = ".png"

# padding color (RGB) - chosen: neutral gray
PADDING_COLOR_RGB = (128, 128, 128)

# ---------------------------
# IMPORTANT: what to generate
# ---------------------------
# If True, the script generates ONLY the pixelation curriculum datasets (easy/hard).
# This is the safest choice when you already generated all other datasets and
# you only want to regenerate pixelation without touching anything else.
PIXELATION_ONLY = True

# If True, delete existing output folders for the things you generate (overwrite).
# If False, script will skip generating a dataset folder if it already exists.
OVERWRITE_EXISTING = True

# Pixelation curriculum presets:
# - pixelation_easy: lighter pixelation factors
# - pixelation_hard: heavier pixelation factors
PIXELATION_PRESETS = {
    "pixelation_easy": [4, 6, 8],
    "pixelation_hard": [10, 12, 16],
}

# If PIXELATION_ONLY = False, the script will also generate these standard degradations
DEGRADATIONS = [
    "blur",
    "noise",
    "low_light",
    "jpeg",
    "pixelation",      # standard mixed pixelation (optional if PIXELATION_ONLY=False)
    "motion_blur",
    "high_light",
    "low_contrast",
    "color_distortion",
]
# =========================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# Resize with aspect ratio + padding (letterbox)
# Input/Output: uint8 RGB
# =========================================================
def resize_with_padding_rgb(
    img: np.ndarray,
    target_hw: Tuple[int, int],
    pad_color_rgb: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """
    Resize preserving aspect ratio, then pad to target size.

    Args:
        img: uint8 RGB (H,W,3)
        target_hw: (H, W)
        pad_color_rgb: (R,G,B) padding color

    Returns:
        uint8 RGB image of shape (target_h, target_w, 3)
    """
    target_h, target_w = target_hw
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero dimension.")

    # scale to fit inside target
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # create canvas
    canvas = np.full((target_h, target_w, 3), pad_color_rgb, dtype=np.uint8)

    # center placement
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    return canvas


# =========================================================
# Degradation functions
# (all operate on uint8 RGB images)
# =========================================================
def blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(img, (k, k), 0)


def noise(img: np.ndarray) -> np.ndarray:
    std = random.uniform(10, 50)
    n = np.random.normal(0, std, img.shape).astype(np.float32)
    out = img.astype(np.float32) + n
    return np.clip(out, 0, 255).astype(np.uint8)


def low_light(img: np.ndarray) -> np.ndarray:
    f = random.uniform(0.05, 0.4)
    out = img.astype(np.float32) * f
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray) -> np.ndarray:
    q = random.randint(10, 50)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    dec = cv2.imdecode(enc, 1)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def pixelation(img: np.ndarray, factor: int) -> np.ndarray:
    """
    Pixelate image by downsampling by 'factor' then upsampling back.
    factor >= 2. Larger factor => stronger pixelation.
    """
    h, w = img.shape[:2]
    # protect against too aggressive factors on small images
    max_factor = max(2, min(h, w) // 2)
    factor = max(2, min(int(factor), max_factor))

    small_w = max(1, w // factor)
    small_h = max(1, h // factor)

    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def pixelation_mixed(img: np.ndarray) -> np.ndarray:
    """
    Standard mixed pixelation used in the original script:
    random factor in [4,16].
    """
    f = random.randint(4, 16)
    return pixelation(img, factor=f)


def motion_blur(img: np.ndarray) -> np.ndarray:
    k = random.randint(5, 25)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def high_light(img: np.ndarray) -> np.ndarray:
    f = random.uniform(1.5, 3.0)
    out = img.astype(np.float32) * f
    return np.clip(out, 0, 255).astype(np.uint8)


def low_contrast(img: np.ndarray) -> np.ndarray:
    a = random.uniform(0.3, 0.7)
    m = img.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    out = a * img.astype(np.float32) + (1 - a) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def color_distortion(img: np.ndarray) -> np.ndarray:
    f = np.random.uniform(0.6, 1.4, size=(1, 1, 3)).astype(np.float32)
    out = img.astype(np.float32) * f
    return np.clip(out, 0, 255).astype(np.uint8)


DEGRADATION_FUNCS = {
    "blur": blur,
    "noise": noise,
    "low_light": low_light,
    "jpeg": jpeg,
    "pixelation": pixelation_mixed,  # standard mixed (only if you want it)
    "motion_blur": motion_blur,
    "high_light": high_light,
    "low_contrast": low_contrast,
    "color_distortion": color_distortion,
}


# =========================================================
# Utilities
# =========================================================
def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_with_padding_rgb(img, TARGET_SIZE, pad_color_rgb=PADDING_COLOR_RGB)
    return img


def save_rgb(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))


def load_or_create_split(files: List[str]) -> Dict[str, List[str]]:
    split_dir = OUTPUT_ROOT / "_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"split_seed{SEED}_test{TEST_RATIO}.json"

    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8"))

    # deterministic split given SEED
    rnd = random.Random(SEED)
    files = files[:]  # copy
    rnd.shuffle(files)

    n_test = int(len(files) * TEST_RATIO)
    split = {
        "train": files[n_test:],
        "test": files[:n_test],
    }

    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def stable_index_from_name(name: str, modulo: int) -> int:
    """
    Stable (cross-run, cross-machine) index based on filename.
    We do NOT use Python's built-in hash() because it's salted per process.
    """
    if modulo <= 0:
        return 0
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


def maybe_prepare_output_dir(base: Path) -> bool:
    """
    Returns True if we should generate into this base folder.
    - If OVERWRITE_EXISTING: delete and recreate
    - Else: skip if exists
    """
    if base.exists():
        if OVERWRITE_EXISTING:
            shutil.rmtree(base)
            return True
        else:
            print(f"[SKIP] '{base}' already exists (OVERWRITE_EXISTING=False)")
            return False
    return True


# =========================================================
# Generation routines
# =========================================================
def generate_standard_degradation(
    degrad: str,
    fn,
    split: Dict[str, List[str]],
) -> None:
    base = OUTPUT_ROOT / degrad
    if not maybe_prepare_output_dir(base):
        return

    for split_name, file_list in split.items():
        for name in tqdm(file_list, desc=f"{degrad} | {split_name}", total=len(file_list)):
            src = CLEAN_DIR / name
            if not src.exists():
                continue

            try:
                img = load_rgb(src)
            except Exception:
                continue

            deg = fn(img)

            out_name = Path(name).stem + OUTPUT_EXT
            save_rgb(img, base / split_name / "clean" / out_name)
            save_rgb(deg, base / split_name / "degraded" / out_name)

    print(f"[OK] Dataset '{degrad}' creato in {base.resolve()}")


def generate_pixelation_curriculum(
    split: Dict[str, List[str]],
) -> None:
    for dataset_name, factors in PIXELATION_PRESETS.items():
        base = OUTPUT_ROOT / dataset_name
        if not maybe_prepare_output_dir(base):
            continue

        for split_name, file_list in split.items():
            for name in tqdm(file_list, desc=f"{dataset_name} | {split_name}", total=len(file_list)):
                src = CLEAN_DIR / name
                if not src.exists():
                    continue

                try:
                    img = load_rgb(src)
                except Exception:
                    continue

                # Deterministic factor choice per image
                idx = stable_index_from_name(name, len(factors))
                factor = factors[idx]

                deg = pixelation(img, factor=factor)

                out_name = Path(name).stem + OUTPUT_EXT
                save_rgb(img, base / split_name / "clean" / out_name)
                save_rgb(deg, base / split_name / "degraded" / out_name)

        print(f"[OK] Dataset '{dataset_name}' creato in {base.resolve()}")


# =========================================================
# Main
# =========================================================
def main():
    clean_imgs = list_images(CLEAN_DIR)
    if not clean_imgs:
        raise RuntimeError(f"No images found in {CLEAN_DIR.resolve()}")

    names = [p.name for p in clean_imgs]
    split = load_or_create_split(names)

    if PIXELATION_ONLY:
        # Only regenerate pixelation curriculum datasets (easy/hard),
        # keeping all other datasets untouched.
        generate_pixelation_curriculum(split)
        print("\n[Done] Generated ONLY pixelation curriculum datasets.")
    else:
        # Generate all standard degradations (+ curriculum if you want)
        for degrad in DEGRADATIONS:
            if degrad not in DEGRADATION_FUNCS:
                raise ValueError(
                    f"Unknown degradation '{degrad}'. Available: {list(DEGRADATION_FUNCS.keys())}"
                )
            fn = DEGRADATION_FUNCS[degrad]
            generate_standard_degradation(degrad, fn, split)

        # Optional: also generate curriculum versions in addition to standard mixed pixelation
        generate_pixelation_curriculum(split)

        print("\n[Done] Generated standard degradations and pixelation curriculum datasets.")

    print(f"\nSplit usato: seed={SEED}, test_ratio={TEST_RATIO}")
    print(f"Resize target: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} (HxW), padding RGB={PADDING_COLOR_RGB}")
    print(f"PIXELATION_ONLY={PIXELATION_ONLY} | OVERWRITE_EXISTING={OVERWRITE_EXISTING}")
    if PIXELATION_PRESETS:
        print(f"Pixelation presets: {PIXELATION_PRESETS}")


if __name__ == "__main__":
    main()
