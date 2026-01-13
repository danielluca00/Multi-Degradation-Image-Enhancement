# data/dataset.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from utils.transforms_factory import (
    build_transforms,
    apply_paired_transform,
    apply_single_transform,
)


def _list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    return sorted([
        f for f in os.listdir(folder)
        if not f.startswith(".") and f.lower().endswith(exts)
    ])


def _stem(filename: str) -> str:
    return os.path.splitext(filename)[0]


class PairedDataset(Dataset):
    """
    Generic paired dataset: input_root (degraded) + target_root (clean).
    Pairing can be:
      - "filename": exact same filename in both folders
      - "stem": same filename without extension (useful if ext differs)
      - "sorted": old behavior (NOT robust) -> sconsigliato
    """
    def __init__(
        self,
        input_root: str,
        target_root: str,
        pairing_mode: str = "filename",
        transform: Optional[Dict] = None,
        image_size: Optional[List[int]] = None,  # kept for backward compatibility
    ):
        super().__init__()
        self.input_root = input_root
        self.target_root = target_root

        inp_files = _list_images(input_root)
        tgt_files = _list_images(target_root)

        if pairing_mode == "sorted":
            # legacy mode
            self.pairs = list(zip(
                [os.path.join(input_root, f) for f in inp_files],
                [os.path.join(target_root, f) for f in tgt_files],
            ))
        else:
            if pairing_mode == "filename":
                inp_map = {f: os.path.join(input_root, f) for f in inp_files}
                tgt_map = {f: os.path.join(target_root, f) for f in tgt_files}
                keys = sorted(set(inp_map.keys()) & set(tgt_map.keys()))
            elif pairing_mode == "stem":
                inp_map = {_stem(f): os.path.join(input_root, f) for f in inp_files}
                tgt_map = {_stem(f): os.path.join(target_root, f) for f in tgt_files}
                keys = sorted(set(inp_map.keys()) & set(tgt_map.keys()))
            else:
                raise ValueError(f"Unknown pairing_mode: {pairing_mode}")

            if len(keys) == 0:
                raise RuntimeError(
                    f"No paired files found with pairing_mode='{pairing_mode}'.\n"
                    f"input_root={input_root}\n"
                    f"target_root={target_root}"
                )

            self.pairs = [(inp_map[k], tgt_map[k]) for k in keys]

        # build transforms from config
        # NOTE: if you want Resize, put it in config ops (preferred)
        self.backend, self.tf = build_transforms(transform, is_paired=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        inp_path, tgt_path = self.pairs[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")

        inp_t, tgt_t = apply_paired_transform(self.backend, self.tf, inp, tgt)
        return inp_t, tgt_t


class UnpairedDataset(Dataset):
    def __init__(
        self,
        input_root: str,
        transform: Optional[Dict] = None,
    ):
        super().__init__()
        self.input_root = input_root
        self.files = [os.path.join(input_root, f) for f in _list_images(input_root)]
        self.backend, self.tf = build_transforms(transform, is_paired=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        inp = Image.open(self.files[idx]).convert("RGB")
        inp_t = apply_single_transform(self.backend, self.tf, inp)
        return inp_t
