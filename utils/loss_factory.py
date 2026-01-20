# utils/loss_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@dataclass
class LossTerm:
    name: str
    weight: float
    mode: str  # "paired" or "unpaired"
    fn: Any    # callable(outputs, targets, inputs) -> torch.Tensor


class LossPipeline:
    """
    Computes a weighted sum of loss terms and also returns individual components.
    """
    def __init__(self, terms: List[LossTerm]):
        self.terms = terms

    def __call__(
        self,
        outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        is_paired: bool = True,
    ) -> Dict[str, torch.Tensor]:
        components: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=outputs.device)

        for term in self.terms:
            if term.mode == "paired" and not is_paired:
                continue
            if term.mode == "unpaired" and is_paired:
                continue

            val = term.fn(outputs=outputs, targets=targets, inputs=inputs)
            # ensure scalar
            if val.ndim != 0:
                val = val.mean()
            components[term.name] = val
            total = total + term.weight * val

        components["total"] = total
        return components


class _VGGPerceptual(nn.Module):
    def __init__(self, layers: int = 20):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:layers]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(x)


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns Sobel kernels of shape [2, 1, 3, 3] for x and y gradients.
    """
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype
    )
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0,  0.0,  0.0],
         [1.0,  2.0,  1.0]],
        device=device, dtype=dtype
    )
    return torch.stack([kx, ky], dim=0).unsqueeze(1)  # [2,1,3,3]


def _image_gradients_sobel(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel gradients for a batch of images.
    Input:  x [B,C,H,W] in [0,1]
    Output: grads [B,C,2,H,W] (2 = dx, dy)
    """
    b, c, h, w = x.shape
    kernels = _sobel_kernels(x.device, x.dtype)  # [2,1,3,3]
    # apply per-channel using groups
    kernels = kernels.repeat(c, 1, 1, 1)  # [2*C,1,3,3]
    x_ = x.view(b * c, 1, h, w)
    g = F.conv2d(x_, kernels, padding=1)  # [B*C, 2, H, W]
    g = g.view(b, c, 2, h, w)
    return g


def build_loss_pipeline(loss_cfg: Optional[Dict[str, Any]], device: str) -> LossPipeline:
    """
    loss_cfg example:
    {
      "enabled": true,
      "terms": [
        {"name": "mse", "weight": 1.0},
        {"name": "vgg_perceptual", "weight": 0.25, "args": {"layers": 20}},
        {"name": "ssim", "weight": 0.5},
        {"name": "lpips", "weight": 0.5, "args": {"net": "alex"}},
        {"name": "gradient_l1", "weight": 0.1, "args": {"to_gray": false}}
      ]
    }
    """
    if not loss_cfg or not loss_cfg.get("enabled", True):
        # default fallback
        loss_cfg = {"terms": [{"name": "mse", "weight": 1.0, "args": {}}]}

    terms_cfg = loss_cfg.get("terms", [])
    if not terms_cfg:
        terms_cfg = [{"name": "mse", "weight": 1.0, "args": {}}]

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    # Lazily created modules
    vgg_module: Optional[_VGGPerceptual] = None
    ssim_metric: Optional[StructuralSimilarityIndexMeasure] = None
    lpips_metric: Optional[LearnedPerceptualImagePatchSimilarity] = None

    built_terms: List[LossTerm] = []

    for t in terms_cfg:
        name = t["name"]
        weight = float(t.get("weight", 1.0))
        args = t.get("args", {}) or {}

        # default mode: paired
        mode = t.get("mode", "paired")

        if name == "mse":
            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("mse loss requires targets (paired dataset).")
                return mse(outputs, targets)
            built_terms.append(LossTerm(name="mse", weight=weight, mode=mode, fn=_fn))

        elif name == "l1":
            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("l1 loss requires targets (paired dataset).")
                return l1(outputs, targets)
            built_terms.append(LossTerm(name="l1", weight=weight, mode=mode, fn=_fn))

        elif name == "charbonnier":
            eps = float(args.get("eps", 1e-3))
            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("charbonnier loss requires targets (paired dataset).")
                diff = outputs - targets
                return torch.mean(torch.sqrt(diff * diff + eps * eps))
            built_terms.append(LossTerm(name="charbonnier", weight=weight, mode=mode, fn=_fn))

        elif name == "vgg_perceptual":
            if vgg_module is None:
                layers = int(args.get("layers", 20))
                vgg_module = _VGGPerceptual(layers=layers).to(device).eval()

            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("vgg_perceptual loss requires targets (paired dataset).")
                return F.mse_loss(vgg_module(outputs), vgg_module(targets))
            built_terms.append(LossTerm(name="vgg_perceptual", weight=weight, mode=mode, fn=_fn))

        elif name == "ssim":
            if ssim_metric is None:
                # expects [0,1] typically; your sigmoid gives that.
                ssim_metric = StructuralSimilarityIndexMeasure().to(device)

            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("ssim loss requires targets (paired dataset).")
                return 1.0 - ssim_metric(outputs, targets)
            built_terms.append(LossTerm(name="ssim", weight=weight, mode=mode, fn=_fn))

        elif name == "lpips":
            if lpips_metric is None:
                net = args.get("net", args.get("net_type", "alex"))
                lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=net).to(device)

            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("lpips loss requires targets (paired dataset).")
                return lpips_metric(outputs, targets)
            built_terms.append(LossTerm(name="lpips", weight=weight, mode=mode, fn=_fn))

        elif name == "gradient_l1":
            # args:
            #  - to_gray: bool (default False)  -> compute gradients on luminance only
            to_gray = bool(args.get("to_gray", False))

            def _to_gray(x: torch.Tensor) -> torch.Tensor:
                # x [B,3,H,W] -> [B,1,H,W]
                if x.shape[1] != 3:
                    return x.mean(dim=1, keepdim=True)
                r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
                return 0.2989 * r + 0.5870 * g + 0.1140 * b

            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("gradient_l1 loss requires targets (paired dataset).")

                x = outputs
                y = targets
                if to_gray:
                    x = _to_gray(x)
                    y = _to_gray(y)

                gx = _image_gradients_sobel(x)  # [B,C,2,H,W]
                gy = _image_gradients_sobel(y)

                # L1 on gradients
                return torch.mean(torch.abs(gx - gy))

            built_terms.append(LossTerm(name="gradient_l1", weight=weight, mode=mode, fn=_fn))

        else:
            raise ValueError(f"Unknown loss term: {name}")

    return LossPipeline(built_terms)
