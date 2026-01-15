# models/model.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from torchvision.transforms import functional as TF

from models.base import BaseModel
from utils.postprocessing_factory import apply_postprocessing

from utils.loss_factory import build_loss_pipeline
from utils.metrics_factory import build_metrics_pipeline


class Model(BaseModel):
    def __init__(self, network, **kwargs):
        super(Model, self).__init__(**kwargs)

        self.network = network.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.scaler = GradScaler()

        # ---- Loss + Metrics from config (fully configurable) ----
        self.loss_cfg = self.config.get("loss", {})
        self.metrics_cfg = self.config.get("metrics", {"enabled": False})

        self.loss_pipe = build_loss_pipeline(self.loss_cfg, device=self.device)
        self.metrics_pipe = build_metrics_pipeline(self.metrics_cfg, device=self.device)

        # ---- Post-processing cfg ----
        self.postproc_cfg = self.config.get("post_processing", {"enabled": False})

        # ---- Output saving cfg ----
        self.save_cfg = self.config.get("save_outputs", {})
        if "output_dir" not in self.save_cfg:
            self.save_cfg["output_dir"] = self.output_images_path

        # saving controls for raw vs postprocessed
        # defaults: if postproc enabled, save postprocessed; raw optional
        self.save_cfg.setdefault("save_raw", False)
        self.save_cfg.setdefault("save_postprocessed", True)

        self.save_cfg.setdefault("raw_prefix", "raw_")
        self.save_cfg.setdefault("post_prefix", self.save_cfg.get("prefix", "output_"))

        # ---- Evaluation controls: compute on raw and/or postprocessed ----
        # default: if postproc enabled, compute both; else raw only
        eval_cfg = self.config.get("evaluation", {})
        self.eval_on_raw = bool(eval_cfg.get("raw", True))
        self.eval_on_post = bool(eval_cfg.get("postprocessed", bool(self.postproc_cfg.get("enabled", False))))

    def _save_batch_outputs(self, outputs: torch.Tensor, start_index: int, prefix: str):
        if not self.save_cfg.get("enabled", True):
            return

        out_dir = self.save_cfg.get("output_dir", "outputs/")
        os.makedirs(out_dir, exist_ok=True)

        resize_hw = self.save_cfg.get("resize_hw", None)  # [h,w] or None
        fmt = self.save_cfg.get("format", "png")

        outputs = outputs.detach().cpu()

        for i in range(outputs.shape[0]):
            img = outputs[i].permute(1, 2, 0).numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)

            if resize_hw is not None:
                img = TF.resize(img, (resize_hw[0], resize_hw[1]))

            path = os.path.join(out_dir, f"{prefix}{start_index + i + 1}.{fmt}")
            img.save(path)

    def _summarize_epoch_components(self, sum_dict: Dict[str, float], denom: int) -> Dict[str, float]:
        if denom <= 0:
            return {k: float("nan") for k in sum_dict.keys()}
        return {k: v / denom for k, v in sum_dict.items()}

    def train_step(self):
        best_loss = float("inf")
        self.network.to(self.device)

        for epoch in range(self.epoch):
            self.network.train()
            epoch_total = 0.0
            comp_sums: Dict[str, float] = {}

            dataloader_iter = tqdm(self.dataloader, desc=f"Training... Epoch: {epoch+1}/{self.epoch}", total=len(self.dataloader))

            for inputs, targets in dataloader_iter:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.network(inputs)

                    loss_dict = self.loss_pipe(
                        outputs=outputs,
                        targets=targets,
                        inputs=inputs,
                        is_paired=True,
                    )
                    loss = loss_dict["total"]

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_total += float(loss.item())
                for k, v in loss_dict.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())

                dataloader_iter.set_postfix({"loss": float(loss.item())})

            epoch_loss = epoch_total / max(1, len(self.dataloader))
            avg_comps = self._summarize_epoch_components(comp_sums, max(1, len(self.dataloader)))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_model(self.network)

            # stampa compatta: total + componenti principali
            comps_str = ", ".join([f"{k}: {avg_comps[k]:.4f}" for k in avg_comps.keys() if k != "total"])
            print(f"Epoch [{epoch+1}/{self.epoch}] Train total: {avg_comps.get('total', epoch_loss):.4f}" + (f" | {comps_str}" if comps_str else ""))

    def test_step(self):
        path = os.path.join(self.model_path, self.model_name)
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()

        out_counter = 0
        max_save = self.save_cfg.get("max_images", None)  # None or int

        # accumulators PRE
        pre_loss_sums: Dict[str, float] = {}
        pre_metric_sums: Dict[str, float] = {}

        # accumulators POST
        post_loss_sums: Dict[str, float] = {}
        post_metric_sums: Dict[str, float] = {}

        n_batches = 0

        with torch.no_grad():
            if self.is_dataset_paired:
                for inputs, targets in tqdm(self.dataloader, desc="Testing..."):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    raw_outputs = self.network(inputs)
                    pp_outputs = apply_postprocessing(raw_outputs, self.postproc_cfg)

                    # ---- PRE (raw) eval ----
                    if self.eval_on_raw:
                        loss_dict = self.loss_pipe(raw_outputs, targets=targets, inputs=inputs, is_paired=True)
                        met_dict = self.metrics_pipe(raw_outputs, targets=targets, inputs=inputs, is_paired=True)

                        for k, v in loss_dict.items():
                            pre_loss_sums[k] = pre_loss_sums.get(k, 0.0) + float(v.item())
                        for k, v in met_dict.items():
                            pre_metric_sums[k] = pre_metric_sums.get(k, 0.0) + float(v.item())

                    # ---- POST (postprocessed) eval ----
                    if self.eval_on_post and self.postproc_cfg.get("enabled", False):
                        loss_dict = self.loss_pipe(pp_outputs, targets=targets, inputs=inputs, is_paired=True)
                        met_dict = self.metrics_pipe(pp_outputs, targets=targets, inputs=inputs, is_paired=True)

                        for k, v in loss_dict.items():
                            post_loss_sums[k] = post_loss_sums.get(k, 0.0) + float(v.item())
                        for k, v in met_dict.items():
                            post_metric_sums[k] = post_metric_sums.get(k, 0.0) + float(v.item())

                    # ---- save images (raw and/or post) ----
                    if self.save_cfg.get("enabled", True) and (max_save is None or out_counter < max_save):
                        if self.save_cfg.get("save_raw", False):
                            self._save_batch_outputs(raw_outputs, start_index=out_counter, prefix=self.save_cfg.get("raw_prefix", "raw_"))

                        if self.save_cfg.get("save_postprocessed", True):
                            # if postproc disabled, pp_outputs == raw_outputs
                            self._save_batch_outputs(pp_outputs, start_index=out_counter, prefix=self.save_cfg.get("post_prefix", "output_"))

                    out_counter += raw_outputs.shape[0]
                    n_batches += 1

                    if max_save is not None and out_counter >= max_save:
                        break

                denom = max(1, n_batches)

                # stampa PRE
                if self.eval_on_raw:
                    pre_loss_avg = {k: v / denom for k, v in pre_loss_sums.items()}
                    pre_met_avg = {k: v / denom for k, v in pre_metric_sums.items()}

                    loss_str = ", ".join([f"{k}: {pre_loss_avg[k]:.4f}" for k in pre_loss_avg.keys()])
                    met_str = ", ".join([f"{k}: {pre_met_avg[k]:.4f}" for k in pre_met_avg.keys()])
                    print(f"[PRE]  Losses -> {loss_str}")
                    if met_str:
                        print(f"[PRE]  Metrics -> {met_str}")

                # stampa POST
                if self.eval_on_post and self.postproc_cfg.get("enabled", False):
                    post_loss_avg = {k: v / denom for k, v in post_loss_sums.items()}
                    post_met_avg = {k: v / denom for k, v in post_metric_sums.items()}

                    loss_str = ", ".join([f"{k}: {post_loss_avg[k]:.4f}" for k in post_loss_avg.keys()])
                    met_str = ", ".join([f"{k}: {post_met_avg[k]:.4f}" for k in post_met_avg.keys()])
                    print(f"[POST] Losses -> {loss_str}")
                    if met_str:
                        print(f"[POST] Metrics -> {met_str}")

            else:
                # unpaired test: only outputs available
                for inputs in tqdm(self.dataloader, desc="Testing..."):
                    inputs = inputs.to(self.device)
                    raw_outputs = self.network(inputs)
                    pp_outputs = apply_postprocessing(raw_outputs, self.postproc_cfg)

                    if self.save_cfg.get("enabled", True) and (max_save is None or out_counter < max_save):
                        if self.save_cfg.get("save_raw", False):
                            self._save_batch_outputs(raw_outputs, start_index=out_counter, prefix=self.save_cfg.get("raw_prefix", "raw_"))
                        if self.save_cfg.get("save_postprocessed", True):
                            self._save_batch_outputs(pp_outputs, start_index=out_counter, prefix=self.save_cfg.get("post_prefix", "output_"))

                    out_counter += raw_outputs.shape[0]
                    n_batches += 1

                    if max_save is not None and out_counter >= max_save:
                        break
