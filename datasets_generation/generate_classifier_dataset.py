# datasets_generation/generate_classifier_dataset.py
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
CLEAN_DIR = Path("clean_images")

# Where we store the classifier dataset
OUTPUT_ROOT = Path("classifier_dataset")

# Final size (H,W) same as enhancers
TARGET_SIZE = (256, 384)
OUTPUT_EXT = ".png"
PADDING_COLOR_RGB = (128, 128, 128)

# Split ratios (must sum to 1.0)
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# How many synthetic variants per clean image
# (Recommended: 3-5. With 2000 clean -> 6000-10000 samples)
VARIANTS_PER_IMAGE = 5

# Multi-degradation: number of degradations per sample
# distribution: 1 (60%), 2 (30%), 3 (10%)
NUM_DEGRADS_CHOICES = [1, 2, 3]
NUM_DEGRADS_PROBS = [0.60, 0.30, 0.10]

# Degradation classes (match your enhancers)
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

# Optionally include "clean-only" samples
INCLUDE_CLEAN_SAMPLES = True
CLEAN_SAMPLE_PROB = 0.10  # 10% samples totally clean (all labels=0)

# Overwrite dataset directory
OVERWRITE_EXISTING = True

# Severity sampling:
# - most of the time: mild severities (Beta(2,5))
# - sometimes: strong severities (Beta(5,2))
HARD_SEV_PROB = 0.15
SEV_BETA_MILD = (2.0, 5.0)
SEV_BETA_HARD = (5.0, 2.0)

# Co-occurrence / realism knobs
# Probability to "encourage" some extra degradations once one is present
# (Not forced, just biases. Keep small to avoid too many unrealistic combos.)
COOCCUR_RULES = {
    # if jpeg chosen, it is relatively common to also have low_contrast or noise
    "jpeg": [("noise", 0.25), ("low_contrast", 0.25)],
    # exposure problems can correlate with low_contrast
    "high_light": [("low_contrast", 0.20)],
    "low_light": [("noise", 0.15), ("low_contrast", 0.20)],
}

# Pipeline templates (order matters)
PIPELINES = [
    # exposure -> blur -> compression -> color/contrast
    ["low_light", "high_light", "blur", "motion_blur", "jpeg", "low_contrast", "color_distortion", "noise", "pixelation"],
    # compression first then noise/blur
    ["jpeg", "pixelation", "noise", "blur", "motion_blur", "low_contrast", "color_distortion", "low_light", "high_light"],
    # color/exposure first then compression
    ["color_distortion", "low_contrast", "low_light", "high_light", "jpeg", "noise", "blur", "motion_blur", "pixelation"],
]
# =========================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =========================================================
# Utils: resize + padding
# =========================================================
def resize_with_padding_rgb(
    img: np.ndarray,
    target_hw: Tuple[int, int],
    pad_color_rgb: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    target_h, target_w = target_hw
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero dimension.")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_color_rgb, dtype=np.uint8)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


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


def md5_int(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16)


def stable_rng(name: str, variant_id: int) -> random.Random:
    # Deterministic RNG per (filename, variant)
    seed_int = (md5_int(f"{name}__v{variant_id}") & 0xFFFFFFFF) ^ SEED
    return random.Random(seed_int)


def stable_np_rng(name: str, variant_id: int) -> np.random.Generator:
    # Deterministic numpy RNG per (filename, variant)
    seed_int = (md5_int(f"np::{name}__v{variant_id}") & 0xFFFFFFFF) ^ SEED
    return np.random.default_rng(seed_int)


# =========================================================
# Split handling (stored inside OUTPUT_ROOT/meta)
# =========================================================
def load_or_create_split(names: List[str]) -> Dict[str, List[str]]:
    if not math.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0, rel_tol=1e-6):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")

    meta_dir = OUTPUT_ROOT / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    split_path = meta_dir / f"split_seed{SEED}_tr{TRAIN_RATIO}_va{VAL_RATIO}_te{TEST_RATIO}.json"

    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8"))

    rnd = random.Random(SEED)
    names = names[:]
    rnd.shuffle(names)

    n = len(names)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    # ensure totals match
    n_test = n - n_train - n_val

    split = {
        "train": names[:n_train],
        "val": names[n_train : n_train + n_val],
        "test": names[n_train + n_val :],
    }

    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


# =========================================================
# Severity sampling
# =========================================================
def sample_severity(rng: random.Random) -> float:
    # Mix of mild and hard severity regimes
    if rng.random() < HARD_SEV_PROB:
        a, b = SEV_BETA_HARD
    else:
        a, b = SEV_BETA_MILD
    # random.Random has betavariate
    sev = rng.betavariate(a, b)
    # clamp for safety
    return float(max(0.0, min(1.0, sev)))


# =========================================================
# Degradation functions (uint8 RGB -> uint8 RGB)
# Each returns: (img_out, severity_used, params_dict)
# =========================================================
def degrade_blur(img: np.ndarray, sev: float, rng: random.Random, np_rng: np.random.Generator):
    # k: 3..9 (odd)
    ks = [3, 5, 7, 9]
    idx = int(round(sev * (len(ks) - 1)))
    k = ks[max(0, min(idx, len(ks) - 1))]
    out = cv2.GaussianBlur(img, (k, k), 0)
    return out, sev, {"k": int(k)}


def degrade_noise(img: np.ndarray, sev: float, rng: random.Random, np_rng: np.random.Generator):
    # std: 5..50
    std = 5.0 + sev * (50.0 - 5.0)
    n = np_rng.normal(0.0, std, img.shape).astype(np.float32)
    out = img.astype(np.float32) + n
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, sev, {"std": float(std)}


def degrade_low_light(img: np.ndarray, sev: float, rng: random.Random, np_rng: np.random.Generator):
    # factor: 0.05..0.45 (sev=1 -> darker)
    factor = 0.45 - sev * (0.45 - 0.05)
    out = img.astype(np.float32) * factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, sev, {"factor": float(factor)}


def degrade_jpeg(img: np.ndarray, sev: float, rng: random.Random, np_rng: np.random.Generator):
    # quality: 10..80 (sev=1 -> low quality)
    quality = int(round(80 - sev * (80 - 10)))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img, sev, {"quality": int(quality), "ok": False}
    dec = cv2.imdecode(enc, 1)
    out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return out, sev, {"quality": int(quality), "ok": True}


def degrade_pixelation(img: np.ndarray, sev: float, rng: random.Random, np_rng: np.random.Generator):
    # factor: 4..16 (sev=1 -> stronger pixelation)
    factor = int(round(4 + sev * (16 - 4)))
    h, w = img.shape[:2]
    factor = max(2, min(factor, min(h, w) // 2))
    small_w = max(1, w // factor)
    small_h = max(1, h // factor)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out, sev, {"factor": int(factor), "small_hw": [int(small_h), int(small_w)]}


def degrade_motion_blur(img: np.ndarray, sev: float, rng: random.Random, np_rng: np.random.Generator):
    # k: 5..25 (sev=1 -> stronger blur)
    k = int(round(5 + sev * (25 - 5)))
    k = max(3, k)
    if k % 2 == 0:
        k += 1

    # random direction (degrees), deterministic thanks to rng
    angle = rng.uniform(0.0, 180.0)

    # build a horizontal line kernel then rotate it
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0

    # rotate around center
    center = (k / 2.0, k / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k), flags=cv2.INTER_LINEAR)

    # normalize (avoid division by zero)
    s = float(kernel.sum())
    if s > 1e-8:
        kernel /= s
    else:
        # fallback: no blur if something goes wrong
        kernel[k // 2, :] = 1.0 / k

    out = cv2.filter2D(img, -1, kernel)
    return out, sev, {"k": int(k), "angle_deg": float(angle)}



def degrade_high_light(img: np.ndarray, sev: float, rng: random.Random, np_rng: np.random.Generator):
    # factor: 1.2..3.0 (sev=1 -> stronger overexposure)
    factor = 1.2 + sev * (3.0 - 1.2)
    out = img.astype(np.float32) * factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, sev, {"factor": float(factor)}


def degrade_low_contrast(img: np.ndarray, sev: float, rng: random.Random, np_rng: np.random.Generator):
    # alpha: 0.2..0.8 (sev=1 -> lower contrast -> smaller alpha)
    alpha = 0.8 - sev * (0.8 - 0.2)
    mean = img.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    out = alpha * img.astype(np.float32) + (1 - alpha) * mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, sev, {"alpha": float(alpha)}


def degrade_color_distortion(img: np.ndarray, sev: float, rng: random.Random, np_rng: np.random.Generator):
    """
    Coherent with your original degradation:
    per-channel gain only (white balance style), then clip.
    Original range: [0.6, 1.4] => +/- 0.4 around 1.0.
    We modulate amplitude with severity.
    """
    amp = 0.4 * sev  # sev=1 -> [0.6, 1.4], sev=0 -> [1.0, 1.0]
    gains = np.array(
        [rng.uniform(1.0 - amp, 1.0 + amp) for _ in range(3)],
        dtype=np.float32
    ).reshape(1, 1, 3)
    out = img.astype(np.float32) * gains
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, sev, {"gains": [float(g) for g in gains.reshape(-1)]}


DEG_FUNCS = {
    "blur": degrade_blur,
    "noise": degrade_noise,
    "low_light": degrade_low_light,
    "jpeg": degrade_jpeg,
    "pixelation": degrade_pixelation,
    "motion_blur": degrade_motion_blur,
    "high_light": degrade_high_light,
    "low_contrast": degrade_low_contrast,
    "color_distortion": degrade_color_distortion,
}


# =========================================================
# Sampling helpers: number of degradations + co-occurrence + pipeline ordering
# =========================================================
def choose_num_degradations(rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for n, p in zip(NUM_DEGRADS_CHOICES, NUM_DEGRADS_PROBS):
        cum += p
        if r <= cum:
            return n
    return NUM_DEGRADS_CHOICES[-1]


def apply_cooccurrence_bias(chosen: List[str], rng: random.Random) -> List[str]:
    chosen_set = set(chosen)
    for d in list(chosen):
        rules = COOCCUR_RULES.get(d, [])
        for other, prob in rules:
            if other in chosen_set:
                continue
            if rng.random() < prob:
                chosen_set.add(other)
    return list(chosen_set)


def order_by_pipeline(chosen: List[str], rng: random.Random) -> List[str]:
    pipeline = rng.choice(PIPELINES)
    rank = {d: i for i, d in enumerate(pipeline)}
    # unknown degradations (should not happen) go last
    return sorted(chosen, key=lambda d: rank.get(d, 10_000))


# =========================================================
# Main generation
# =========================================================
def main():
    if OVERWRITE_EXISTING and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    clean_imgs = list_images(CLEAN_DIR)
    if not clean_imgs:
        raise RuntimeError(f"No images found in {CLEAN_DIR.resolve()}")

    names = [p.name for p in clean_imgs]
    split = load_or_create_split(names)

    # Save classes list
    meta_dir = OUTPUT_ROOT / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "classes.json").write_text(json.dumps(DEGRADATIONS, indent=2), encoding="utf-8")
    (meta_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "target_size_hw": list(TARGET_SIZE),
                "variants_per_image": VARIANTS_PER_IMAGE,
                "include_clean_samples": INCLUDE_CLEAN_SAMPLES,
                "clean_sample_prob": CLEAN_SAMPLE_PROB,
                "num_degrads_choices": NUM_DEGRADS_CHOICES,
                "num_degrads_probs": NUM_DEGRADS_PROBS,
                "hard_sev_prob": HARD_SEV_PROB,
                "sev_beta_mild": list(SEV_BETA_MILD),
                "sev_beta_hard": list(SEV_BETA_HARD),
                "cooccur_rules": COOCCUR_RULES,
                "pipelines": PIPELINES,
                "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Create splits
    for split_name, file_list in split.items():
        out_img_dir = OUTPUT_ROOT / split_name / "images"
        out_lbl_path = OUTPUT_ROOT / split_name / "labels.jsonl"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        total_samples = len(file_list) * VARIANTS_PER_IMAGE

        with out_lbl_path.open("w", encoding="utf-8") as f:
            pbar = tqdm(total=total_samples, desc=f"Generating classifier split={split_name}")
            for name in file_list:
                src = CLEAN_DIR / name
                if not src.exists():
                    pbar.update(VARIANTS_PER_IMAGE)
                    continue

                try:
                    base_img = load_rgb(src)
                except Exception:
                    pbar.update(VARIANTS_PER_IMAGE)
                    continue

                for variant_id in range(VARIANTS_PER_IMAGE):
                    rng = stable_rng(name, variant_id)
                    np_rng = stable_np_rng(name, variant_id)

                    labels = {c: 0 for c in DEGRADATIONS}
                    severity = {c: 0.0 for c in DEGRADATIONS}
                    params: Dict[str, Dict] = {}

                    # clean-only sample?
                    if INCLUDE_CLEAN_SAMPLES and rng.random() < CLEAN_SAMPLE_PROB:
                        out = base_img
                        chosen: List[str] = []
                    else:
                        n_deg = choose_num_degradations(rng)
                        chosen = rng.sample(DEGRADATIONS, k=n_deg)

                        # co-occurrence bias (may increase count beyond n_deg)
                        chosen = apply_cooccurrence_bias(chosen, rng)

                        # realistic ordering via pipeline template
                        chosen = order_by_pipeline(chosen, rng)

                        out = base_img
                        for d in chosen:
                            sev = sample_severity(rng)
                            out, used, p = DEG_FUNCS[d](out, sev, rng, np_rng)
                            labels[d] = 1
                            severity[d] = float(used)
                            params[d] = p

                    # filename with variant
                    stem = Path(name).stem
                    out_name = f"{stem}__v{variant_id}{OUTPUT_EXT}"
                    save_rgb(out, out_img_dir / out_name)

                    rec = {
                        "file": str(Path(split_name) / "images" / out_name),
                        "source_clean": name,
                        "variant_id": int(variant_id),
                        "chosen_degradations": chosen,
                        "labels": labels,
                        "severity": severity,
                        "params": params,
                    }
                    f.write(json.dumps(rec) + "\n")
                    pbar.update(1)

            pbar.close()

    # Print summary
    n_total = len(names) * VARIANTS_PER_IMAGE
    n_train = len(split["train"]) * VARIANTS_PER_IMAGE
    n_val = len(split["val"]) * VARIANTS_PER_IMAGE
    n_test = len(split["test"]) * VARIANTS_PER_IMAGE

    print("\n[OK] Classifier dataset generated at:", OUTPUT_ROOT.resolve())
    print("Classes:", DEGRADATIONS)
    print(f"Target size (H,W): {TARGET_SIZE} | seed={SEED}")
    print(f"Variants per image: {VARIANTS_PER_IMAGE}")
    print(f"Total samples: {n_total}  (train={n_train}, val={n_val}, test={n_test})")
    print("Splits stored in:", (OUTPUT_ROOT / "meta").resolve())


if __name__ == "__main__":
    main()
