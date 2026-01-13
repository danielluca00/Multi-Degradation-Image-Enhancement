# models/model.py
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import torchvision.models as models
from torchvision.transforms import functional as TF

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from torch.cuda.amp import autocast, GradScaler

from models.base import BaseModel
from utils.postprocessing_factory import apply_postprocessing


class Model(BaseModel):
    def __init__(self, network, **kwargs):
        super(Model, self).__init__(**kwargs)

        self.network = network.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.scaler = GradScaler()

        # ---- Loss cfg from config ----
        loss_cfg = self.config.get("loss", {})
        self.lambda_mse = float(loss_cfg.get("lambda_mse", 1.0))
        self.lambda_vgg = float(loss_cfg.get("lambda_vgg", 0.25))
        self.lambda_ssim = float(loss_cfg.get("lambda_ssim", 0.5))
        self.lambda_lpips = float(loss_cfg.get("lambda_lpips", 0.5))

        self.criterion = nn.MSELoss()

        # Metrics/loss components (instantiate only if needed)
        self.ssim_metric = None
        self.lpips_metric = None
        self.vgg = None

        if self.lambda_ssim > 0:
            self.ssim_metric = StructuralSimilarityIndexMeasure().to(self.device)

        if self.lambda_lpips > 0:
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=loss_cfg.get("lpips_net", "alex")).to(self.device)

        if self.lambda_vgg > 0:
            vgg_layers = int(loss_cfg.get("vgg_layers", 20))
            self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:vgg_layers].to(self.device)
            for p in self.vgg.parameters():
                p.requires_grad = False

        # ---- Post-processing cfg ----
        self.postproc_cfg = self.config.get("post_processing", {"enabled": False})

        # ---- Output saving cfg ----
        self.save_cfg = self.config.get("save_outputs", {})
        # fallback to test.output_images_path if user keeps old field
        if "output_dir" not in self.save_cfg:
            self.save_cfg["output_dir"] = self.output_images_path

    def composite_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = 0.0

        if self.lambda_mse > 0:
            total = total + self.lambda_mse * self.criterion(outputs, targets)

        if self.lambda_vgg > 0 and self.vgg is not None:
            total = total + self.lambda_vgg * F.mse_loss(self.vgg(outputs), self.vgg(targets))

        if self.lambda_ssim > 0 and self.ssim_metric is not None:
            total = total + self.lambda_ssim * (1.0 - self.ssim_metric(outputs, targets))

        if self.lambda_lpips > 0 and self.lpips_metric is not None:
            total = total + self.lambda_lpips * self.lpips_metric(outputs, targets)

        return total

    def _save_batch_outputs(self, outputs: torch.Tensor, start_index: int):
        if not self.save_cfg.get("enabled", True):
            return

        out_dir = self.save_cfg.get("output_dir", "outputs/")
        os.makedirs(out_dir, exist_ok=True)

        resize_hw = self.save_cfg.get("resize_hw", None)  # [h,w] or None
        fmt = self.save_cfg.get("format", "png")
        prefix = self.save_cfg.get("prefix", "output_")

        outputs = outputs.detach().cpu()

        for i in range(outputs.shape[0]):
            img = outputs[i].permute(1, 2, 0).numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)

            if resize_hw is not None:
                img = TF.resize(img, (resize_hw[0], resize_hw[1]))

            path = os.path.join(out_dir, f"{prefix}{start_index + i + 1}.{fmt}")
            img.save(path)

    def train_step(self):
        best_loss = float("inf")
        self.network.to(self.device)

        for epoch in range(self.epoch):
            self.network.train()
            epoch_loss = 0.0

            dataloader_iter = tqdm(self.dataloader, desc=f"Training... Epoch: {epoch+1}/{self.epoch}", total=len(self.dataloader))

            for inputs, targets in dataloader_iter:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.network(inputs)
                    loss = self.composite_loss(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                dataloader_iter.set_postfix({"loss": loss.item()})

            epoch_loss /= len(self.dataloader)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_model(self.network)

            print(f"Epoch [{epoch+1}/{self.epoch}] Train Loss: {epoch_loss:.4f}")

    def test_step(self):
        path = os.path.join(self.model_path, self.model_name)
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()

        # metrics
        psnr = PeakSignalNoiseRatio().to(self.device)
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type=self.config.get("loss", {}).get("lpips_net", "alex")).to(self.device)

        out_counter = 0
        max_save = self.save_cfg.get("max_images", None)  # None or int

        test_loss = 0.0
        test_psnr = 0.0
        test_ssim = 0.0
        test_lpips = 0.0

        with torch.no_grad():
            if self.is_dataset_paired:
                for inputs, targets in tqdm(self.dataloader, desc="Testing..."):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.network(inputs)

                    # post-processing fully controlled by config
                    outputs = apply_postprocessing(outputs, self.postproc_cfg)

                    loss = self.composite_loss(outputs, targets)
                    test_loss += loss.item()
                    test_psnr += psnr(outputs, targets)
                    test_ssim += ssim(outputs, targets)
                    test_lpips += lpips(outputs, targets)

                    # save outputs (optionally limited)
                    if max_save is None or out_counter < max_save:
                        self._save_batch_outputs(outputs, start_index=out_counter)
                    out_counter += outputs.shape[0]

                    if max_save is not None and out_counter >= max_save:
                        break

                n = max(1, len(self.dataloader))
                test_loss /= n
                test_psnr /= n
                test_ssim /= n
                test_lpips /= n

                print(
                    f"Test Loss: {test_loss:.4f}, "
                    f"Test PSNR: {test_psnr:.4f}, "
                    f"Test SSIM: {test_ssim:.4f}, "
                    f"Test LPIPS: {test_lpips:.4f}"
                )

            else:
                for inputs in tqdm(self.dataloader, desc="Testing..."):
                    inputs = inputs.to(self.device)
                    outputs = self.network(inputs)
                    outputs = apply_postprocessing(outputs, self.postproc_cfg)

                    if max_save is None or out_counter < max_save:
                        self._save_batch_outputs(outputs, start_index=out_counter)
                    out_counter += outputs.shape[0]

                    if max_save is not None and out_counter >= max_save:
                        break
