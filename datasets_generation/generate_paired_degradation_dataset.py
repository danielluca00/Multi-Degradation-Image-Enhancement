# generate_paired_degradation_dataset.py
from __future__ import annotations

import json
import random
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

DEGRADATIONS = [
    "blur",
    "noise",
    "low_light",
    "jpeg",
    "pixelation",
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
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

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

def pixelation(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    f = random.randint(4, 16)
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

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
    "pixelation": pixelation,
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


# =========================================================
# Main
# =========================================================
def main():
    clean_imgs = list_images(CLEAN_DIR)
    if not clean_imgs:
        raise RuntimeError(f"No images found in {CLEAN_DIR.resolve()}")

    names = [p.name for p in clean_imgs]
    split = load_or_create_split(names)

    for degrad in DEGRADATIONS:
        if degrad not in DEGRADATION_FUNCS:
            raise ValueError(f"Unknown degradation '{degrad}'. Available: {list(DEGRADATION_FUNCS.keys())}")

        fn = DEGRADATION_FUNCS[degrad]
        base = OUTPUT_ROOT / degrad

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

    print("\nTutti i dataset paired creati con successo.")
    print(f"Split usato: seed={SEED}, test_ratio={TEST_RATIO}")
    print(f"Resize target: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} (HxW), padding RGB={PADDING_COLOR_RGB}")


if __name__ == "__main__":
    main()
