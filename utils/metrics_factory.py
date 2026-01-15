# utils/metrics_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@dataclass
class MetricItem:
    name: str
    mode: str  # "paired" or "unpaired"
    fn: Any    # callable(outputs, targets, inputs) -> torch.Tensor


class MetricsPipeline:
    def __init__(self, metrics: Dict[str, MetricItem]):
        self.metrics = metrics

    def __call__(
        self,
        outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        is_paired: bool = True,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, item in self.metrics.items():
            if item.mode == "paired" and not is_paired:
                continue
            if item.mode == "unpaired" and is_paired:
                continue
            val = item.fn(outputs=outputs, targets=targets, inputs=inputs)
            if val.ndim != 0:
                val = val.mean()
            out[name] = val
        return out


def build_metrics_pipeline(metrics_cfg: Optional[Dict[str, Any]], device: str) -> MetricsPipeline:
    """
    metrics_cfg example:
    {
      "enabled": true,
      "items": [
        {"name": "psnr"},
        {"name": "ssim"},
        {"name": "lpips", "args": {"net": "alex"}}
      ]
    }
    """
    if not metrics_cfg or not metrics_cfg.get("enabled", True):
        # default: no metrics
        return MetricsPipeline({})

    items = metrics_cfg.get("items", [])
    metrics: Dict[str, MetricItem] = {}

    # instantiate metric modules lazily
    psnr_mod: Optional[PeakSignalNoiseRatio] = None
    ssim_mod: Optional[StructuralSimilarityIndexMeasure] = None
    lpips_mod: Optional[LearnedPerceptualImagePatchSimilarity] = None

    for it in items:
        name = it["name"]
        args = it.get("args", {}) or {}
        mode = it.get("mode", "paired")

        if name == "psnr":
            if psnr_mod is None:
                psnr_mod = PeakSignalNoiseRatio().to(device)

            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("psnr metric requires targets (paired dataset).")
                return psnr_mod(outputs, targets)

            metrics["psnr"] = MetricItem(name="psnr", mode=mode, fn=_fn)

        elif name == "ssim":
            if ssim_mod is None:
                ssim_mod = StructuralSimilarityIndexMeasure().to(device)

            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("ssim metric requires targets (paired dataset).")
                return ssim_mod(outputs, targets)

            metrics["ssim"] = MetricItem(name="ssim", mode=mode, fn=_fn)

        elif name == "lpips":
            if lpips_mod is None:
                net = args.get("net", args.get("net_type", "alex"))
                lpips_mod = LearnedPerceptualImagePatchSimilarity(net_type=net).to(device)

            def _fn(outputs, targets, inputs=None):
                if targets is None:
                    raise ValueError("lpips metric requires targets (paired dataset).")
                return lpips_mod(outputs, targets)

            metrics["lpips"] = MetricItem(name="lpips", mode=mode, fn=_fn)

        else:
            raise ValueError(f"Unknown metric: {name}")

    return MetricsPipeline(metrics)
