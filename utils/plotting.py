# utils/plotting.py
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_epoch_rows(train_csv_path: str) -> Tuple[List[int], List[Dict[str, float]]]:
    epochs, rows = [], []

    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return epochs, rows

        has_type = "type" in reader.fieldnames

        for r in reader:
            if has_type and r.get("type") != "epoch":
                continue
            if "epoch" not in r or not r["epoch"]:
                continue

            epoch = int(float(r["epoch"]))
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass

            epochs.append(epoch)
            rows.append(row)

    return epochs, rows


def plot_losses_from_csv(train_csv: str, out_dir: str) -> None:
    epochs, rows = _read_epoch_rows(train_csv)
    if not epochs:
        return

    os.makedirs(out_dir, exist_ok=True)

    # collect loss_* columns
    keys = sorted({k for r in rows for k in r if k.startswith("loss_")})
    if not keys:
        return

    series = {k: [r.get(k, float("nan")) for r in rows] for k in keys}

    # 1) total loss
    if "loss_total" in series:
        _plot_single(epochs, series["loss_total"], "loss_total", os.path.join(out_dir, "loss_total.png"))

    # 2) each component
    for k, y in series.items():
        if k != "loss_total":
            _plot_single(epochs, y, k, os.path.join(out_dir, f"{k}.png"))

    # 3) all together
    _plot_multi(epochs, series, os.path.join(out_dir, "loss_all.png"))


def _plot_single(x, y, title, path):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_multi(x, series, path):
    plt.figure()
    for name, y in series.items():
        plt.plot(x, y, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.title("Loss curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
