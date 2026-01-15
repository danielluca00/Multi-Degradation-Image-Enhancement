# models/model.py
from __future__ import annotations

import os
import shutil
import time
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

        self.save_cfg.setdefault("save_raw", False)
        self.save_cfg.setdefault("save_postprocessed", True)
        self.save_cfg.setdefault("raw_prefix", "raw_")
        self.save_cfg.setdefault("post_prefix", self.save_cfg.get("prefix", "output_"))

        # ---- Evaluation controls: compute on raw and/or postprocessed ----
        eval_cfg = self.config.get("evaluation", {})
        self.eval_on_raw = bool(eval_cfg.get("raw", True))
        self.eval_on_post = bool(eval_cfg.get("postprocessed", bool(self.postproc_cfg.get("enabled", False))))

        # ---- Logging cfg ----
        log_cfg = self.config.get("logging", {}) or {}
        self.logging_enabled = bool(log_cfg.get("enabled", False))
        self.train_log_every = int((log_cfg.get("train", {}) or {}).get("log_every_n_batches", 0) or 0)

        # ---- Checkpoints cfg ----
        ckpt_cfg = log_cfg.get("checkpoints", {}) or {}
        self.ckpt_enabled = bool(ckpt_cfg.get("enabled", False))
        self.ckpt_every = int(ckpt_cfg.get("every_n_epochs", 10))

        # best tracking
        self.best_loss = float("inf")

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

    def _maybe_save_epoch_checkpoint(self, epoch_idx_0based: int):
        """
        Saves an extra checkpoint (optional) into the run directory:
        runs/<task>/<timestamp>/checkpoints/epoch_XX.pt
        """
        if not (self.logging_enabled and self.ckpt_enabled and self.logger is not None):
            return
        if self.ckpt_every <= 0:
            return
        epoch_num = epoch_idx_0based + 1
        if epoch_num % self.ckpt_every != 0:
            return

        run_dir = getattr(self.logger, "run_dir", lambda: None)()
        if not run_dir:
            return

        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch_num:03d}.pt")
        torch.save(self.network.state_dict(), ckpt_path)

    def _maybe_copy_best_to_run_dir(self):
        """
        Optional convenience: copy best weights file into run dir as best.pt
        (only if logger is enabled and run_dir exists).
        """
        if not (self.logging_enabled and self.logger is not None):
            return
        run_dir = getattr(self.logger, "run_dir", lambda: None)()
        if not run_dir:
            return
        src = os.path.join(self.model_path, self.model_name)
        if os.path.isfile(src):
            dst = os.path.join(run_dir, "best.pt")
            try:
                shutil.copyfile(src, dst)
            except Exception:
                pass

    def train_step(self):
        self.network.to(self.device)

        for epoch in range(self.epoch):
            t0 = time.time()

            self.network.train()
            epoch_total = 0.0
            comp_sums: Dict[str, float] = {}

            dataloader_iter = tqdm(
                enumerate(self.dataloader),
                desc=f"Training... Epoch: {epoch+1}/{self.epoch}",
                total=len(self.dataloader),
            )

            for step, batch in dataloader_iter:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.network(inputs)
                    loss_dict = self.loss_pipe(outputs=outputs, targets=targets, inputs=inputs, is_paired=True)
                    loss = loss_dict["total"]

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_total += float(loss.item())
                for k, v in loss_dict.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())

                dataloader_iter.set_postfix({"loss": float(loss.item())})

                # ---- optional batch-level logging ----
                if self.logging_enabled and self.logger is not None and self.train_log_every > 0:
                    if (step + 1) % self.train_log_every == 0:
                        row = {
                            "type": "batch",
                            "epoch": epoch + 1,
                            "step": step + 1,
                        }
                        # log total + components (current batch)
                        for k, v in loss_dict.items():
                            row[f"loss_{k}"] = float(v.item())
                        self.logger.log_train(row)

            denom = max(1, len(self.dataloader))
            avg_comps = self._summarize_epoch_components(comp_sums, denom)
            epoch_loss = float(avg_comps.get("total", epoch_total / denom))

            # ---- best model ----
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_model(self.network)
                self._maybe_copy_best_to_run_dir()

            # ---- epoch-level logging ----
            if self.logging_enabled and self.logger is not None:
                epoch_time = time.time() - t0
                row = {
                    "type": "epoch",
                    "epoch": epoch + 1,
                    "epoch_time_sec": float(epoch_time),
                    "lr": float(self.lr),
                    "best_loss_so_far": float(self.best_loss),
                }
                for k, v in avg_comps.items():
                    row[f"loss_{k}"] = float(v)
                self.logger.log_train(row)

                # update summary incrementally
                self.logger.set_summary({
                    "best_train_loss": float(self.best_loss),
                    "epochs_completed": int(epoch + 1),
                })

            # ---- optional periodic checkpoints ----
            self._maybe_save_epoch_checkpoint(epoch_idx_0based=epoch)

            # console print
            comps_str = ", ".join([f"{k}: {avg_comps[k]:.4f}" for k in avg_comps.keys() if k != "total"])
            print(
                f"Epoch [{epoch+1}/{self.epoch}] "
                f"Train total: {avg_comps.get('total', epoch_loss):.4f}"
                + (f" | {comps_str}" if comps_str else "")
                + f" | best: {self.best_loss:.4f}"
            )

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
                            self._save_batch_outputs(pp_outputs, start_index=out_counter, prefix=self.save_cfg.get("post_prefix", "output_"))

                    out_counter += raw_outputs.shape[0]
                    n_batches += 1

                    if max_save is not None and out_counter >= max_save:
                        break

                denom = max(1, n_batches)

                pre_loss_avg = {k: v / denom for k, v in pre_loss_sums.items()}
                pre_met_avg = {k: v / denom for k, v in pre_metric_sums.items()}

                post_loss_avg = {k: v / denom for k, v in post_loss_sums.items()}
                post_met_avg = {k: v / denom for k, v in post_metric_sums.items()}

                # stampa PRE
                if self.eval_on_raw:
                    loss_str = ", ".join([f"{k}: {pre_loss_avg[k]:.4f}" for k in pre_loss_avg.keys()])
                    met_str = ", ".join([f"{k}: {pre_met_avg[k]:.4f}" for k in pre_met_avg.keys()])
                    print(f"[PRE]  Losses -> {loss_str}")
                    if met_str:
                        print(f"[PRE]  Metrics -> {met_str}")

                # stampa POST
                if self.eval_on_post and self.postproc_cfg.get("enabled", False):
                    loss_str = ", ".join([f"{k}: {post_loss_avg[k]:.4f}" for k in post_loss_avg.keys()])
                    met_str = ", ".join([f"{k}: {post_met_avg[k]:.4f}" for k in post_met_avg.keys()])
                    print(f"[POST] Losses -> {loss_str}")
                    if met_str:
                        print(f"[POST] Metrics -> {met_str}")

                # ---- logging test rows ----
                if self.logging_enabled and self.logger is not None:
                    if self.eval_on_raw:
                        row = {"type": "test", "stage": "pre", "batches": int(n_batches)}
                        for k, v in pre_loss_avg.items():
                            row[f"loss_{k}"] = float(v)
                        for k, v in pre_met_avg.items():
                            row[f"metric_{k}"] = float(v)
                        self.logger.log_test(row)

                    if self.eval_on_post and self.postproc_cfg.get("enabled", False):
                        row = {"type": "test", "stage": "post", "batches": int(n_batches)}
                        for k, v in post_loss_avg.items():
                            row[f"loss_{k}"] = float(v)
                        for k, v in post_met_avg.items():
                            row[f"metric_{k}"] = float(v)
                        self.logger.log_test(row)

                    # update summary
                    self.logger.set_summary({
                        "best_train_loss": float(self.best_loss),
                        "test_batches": int(n_batches),
                        "post_processing_enabled": bool(self.postproc_cfg.get("enabled", False)),
                    })

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

                if self.logging_enabled and self.logger is not None:
                    self.logger.log_test({"type": "test", "stage": "unpaired", "batches": int(n_batches)})
                    self.logger.set_summary({
                        "best_train_loss": float(self.best_loss),
                        "test_batches": int(n_batches),
                        "post_processing_enabled": bool(self.postproc_cfg.get("enabled", False)),
                    })
