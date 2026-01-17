# select_clean_images_imagenet.py
from __future__ import annotations

import random
import shutil
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
IMAGENET_DIR = Path(r"C:\Users\dani2\Downloads\archive")    # cartella con sottocartelle = classi
OUTPUT_DIR = Path("clean_images")                           # cartella di output (flat)
NUM_CLASSES = 100                                           # numero classi da selezionare
IMAGES_PER_CLASS = 20                                       # immagini per classe
SEED = 42                                                   # seed per riproducibilità
FLAT_OUTPUT = True                                          # True = tutte le immagini in una cartella
# =========================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_classes(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_images(cls_dir: Path):
    return sorted([
        p for p in cls_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])


def main():
    classes = list_classes(IMAGENET_DIR)
    if not classes:
        raise RuntimeError(f"Nessuna classe trovata in {IMAGENET_DIR}")

    random.shuffle(classes)
    selected_classes = classes[:NUM_CLASSES]

    copied = 0
    for cls in selected_classes:
        images = list_images(cls)
        if not images:
            continue

        random.shuffle(images)
        images = images[:IMAGES_PER_CLASS]

        cls_name = cls.name
        if not FLAT_OUTPUT:
            (OUTPUT_DIR / cls_name).mkdir(parents=True, exist_ok=True)

        for img in images:
            # nome unico per evitare collisioni
            new_name = f"{cls_name}__{img.name}"
            dst = OUTPUT_DIR / new_name if FLAT_OUTPUT else OUTPUT_DIR / cls_name / img.name
            shutil.copy2(img, dst)
            copied += 1

    print(f"[OK] Copiate {copied} immagini in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
