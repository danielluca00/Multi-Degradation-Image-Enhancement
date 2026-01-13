# utils/postprocessing_factory.py
from __future__ import annotations

from typing import Any, Dict

import torch

from utils.post_processing import enhance_contrast, enhance_color, sharpen, soft_denoise


_OPS = {
    "enhance_contrast": enhance_contrast,
    "enhance_color": enhance_color,
    "sharpen": sharpen,
    "soft_denoise": soft_denoise,
}


def apply_postprocessing(images: torch.Tensor, pp_cfg: Dict[str, Any]) -> torch.Tensor:
    """
    pp_cfg example:
    {
      "enabled": true,
      "ops": [
        {"name": "enhance_contrast", "args": {"contrast_factor": 1.03}},
        {"name": "enhance_color", "args": {"saturation_factor": 1.55}}
      ]
    }
    """
    if not pp_cfg or not pp_cfg.get("enabled", False):
        return images

    ops = pp_cfg.get("ops", [])
    out = images
    for op in ops:
        name = op["name"]
        args = op.get("args", {})
        if name not in _OPS:
            raise ValueError(f"Unknown post-processing op: {name}")
        out = _OPS[name](out, **args)
    return out
