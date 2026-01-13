# models/base.py
from __future__ import annotations

import os
import time
from abc import abstractmethod

import torch


class BaseModel:
    def __init__(self, config, dataloader):
        self.config = config
        self.phase = config["phase"]

        self.device = config[self.phase]["device"]
        self.batch_size = config[self.phase]["dataloader"]["args"]["batch_size"]
        self.epoch = config["train"]["n_epoch"]
        self.lr = config["train"]["lr"]

        test_cfg = config.get("test", {})
        test_dataset_cfg = test_cfg.get("dataset", {})

        self.is_dataset_paired = bool(test_dataset_cfg.get("is_paired", True))
        self.dataloader = dataloader

        self.model_path = config[self.phase]["model_path"]
        self.model_name = config[self.phase]["model_name"]

        # backward compatible field
        self.output_images_path = test_cfg.get("output_images_path", "outputs/")

    def train(self):
        since = time.time()
        self.train_step()
        t = time.time() - since
        print(f"Training completed in {t//60:.0f}m {t%60:.0f}s")

    def test(self):
        self.test_step()

    @abstractmethod
    def train_step(self):
        raise NotImplementedError

    @abstractmethod
    def val_step(self):
        raise NotImplementedError

    def save_model(self, model):
        os.makedirs(self.model_path, exist_ok=True)
        path = os.path.join(self.model_path, self.model_name)
        torch.save(model.state_dict(), path)
