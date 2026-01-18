# utils/transforms_factory.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

import torch

# torchvision
from torchvision import transforms as T

# albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _tv_build(ops):
    tfs = []
    for op in ops:
        name = op["name"]
        args = op.get("args", {})

        if name == "Resize":
            # support both {"height":..,"width":..} and {"size":[h,w]}
            if "size" in args:
                size = tuple(args["size"])
            else:
                size = (args["height"], args["width"])
            tfs.append(T.Resize(size))
        elif name == "ToTensor":
            tfs.append(T.ToTensor())
        elif name == "Normalize":
            tfs.append(T.Normalize(mean=args["mean"], std=args["std"]))
        elif name == "ColorJitter":
            tfs.append(T.ColorJitter(**args))
        elif name == "RandomHorizontalFlip":
            tfs.append(T.RandomHorizontalFlip(p=args.get("p", 0.5)))
        elif name == "RandomVerticalFlip":
            tfs.append(T.RandomVerticalFlip(p=args.get("p", 0.5)))
        elif name == "RandomRotation":
            tfs.append(T.RandomRotation(degrees=args.get("degrees", 0)))
        else:
            raise ValueError(f"[torchvision] Transform not supported: {name}")

    return T.Compose(tfs)


def _albu_build(ops, is_paired: bool):
    albu_ops = []
    for op in ops:
        name = op["name"]
        args = op.get("args", {})

        if name == "Resize":
            albu_ops.append(A.Resize(height=args["height"], width=args["width"]))
        elif name == "HorizontalFlip":
            albu_ops.append(A.HorizontalFlip(p=args.get("p", 0.5)))
        elif name == "VerticalFlip":
            albu_ops.append(A.VerticalFlip(p=args.get("p", 0.5)))
        elif name == "RandomRotate90":
            albu_ops.append(A.RandomRotate90(p=args.get("p", 0.5)))
        elif name == "RandomBrightnessContrast":
            albu_ops.append(A.RandomBrightnessContrast(**args))
        elif name == "GaussNoise":
            albu_ops.append(A.GaussNoise(**args))
        elif name == "MotionBlur":
            albu_ops.append(A.MotionBlur(**args))
        elif name == "HueSaturationValue":
            albu_ops.append(A.HueSaturationValue(**args))
        elif name == "RandomGamma":
            albu_ops.append(A.RandomGamma(**args))
        elif name == "CLAHE":
            albu_ops.append(A.CLAHE(**args))
        elif name == "Sharpen":
            albu_ops.append(A.Sharpen(**args))
        elif name == "Normalize":
            albu_ops.append(A.Normalize(mean=args["mean"], std=args["std"]))
        elif name == "ToTensorV2":
            albu_ops.append(ToTensorV2())
        else:
            raise ValueError(f"[albumentations] Transform not supported: {name}")

    additional_targets = {"target": "image"} if is_paired else None
    return A.Compose(albu_ops, additional_targets=additional_targets)


def build_transforms(transform_cfg: Optional[Dict[str, Any]], is_paired: bool):
    """
    Returns:
      - backend: "torchvision" or "albumentations"
      - transform object
    """
    if not transform_cfg:
        # default: just ToTensor
        return "torchvision", T.ToTensor()

    backend = transform_cfg.get("backend", "torchvision")
    ops = transform_cfg.get("ops", [])

    if backend == "torchvision":
        return backend, _tv_build(ops)

    if backend == "albumentations":
        return backend, _albu_build(ops, is_paired=is_paired)

    raise ValueError(f"Unknown transform backend: {backend}")


def apply_paired_transform(backend: str, tf, inp_pil: Image.Image, tgt_pil: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    if backend == "albumentations":
        inp = np.array(inp_pil)
        tgt = np.array(tgt_pil)
        out = tf(image=inp, target=tgt)
        return out["image"], out["target"]

    # torchvision: apply separately (OK for deterministic ops, not ideal for random augmentations)
    return tf(inp_pil), tf(tgt_pil)


def apply_single_transform(backend: str, tf, inp_pil: Image.Image) -> torch.Tensor:
    if backend == "albumentations":
        inp = np.array(inp_pil)
        out = tf(image=inp)
        return out["image"]
    return tf(inp_pil)
