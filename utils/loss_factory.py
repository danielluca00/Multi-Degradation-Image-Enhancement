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


def build_loss_pipeline(loss_cfg: Optional[Dict[str, Any]], device: str) -> LossPipeline:
    """
    loss_cfg example:
    {
      "enabled": true,
      "terms": [
        {"name": "mse", "weight": 1.0},
        {"name": "vgg_perceptual", "weight": 0.25, "args": {"layers": 20}},
        {"name": "ssim", "weight": 0.5},
        {"name": "lpips", "weight": 0.5, "args": {"net": "alex"}}
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

        else:
            raise ValueError(f"Unknown loss term: {name}")

    return LossPipeline(built_terms)
