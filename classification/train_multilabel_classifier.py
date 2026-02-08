# classification/train_multilabel_classifier.py
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
DATASET_ROOT = Path("classifier_dataset")
RUN_BASE = Path("runs_classifier")

BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4
PATIENCE = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# threshold for sigmoid -> label
THRESH = 0.5

# loss weights
LAMBDA_SEVERITY = 0.5  # weight of severity loss

# dataloader
NUM_WORKERS = 4
PIN_MEMORY = True

# =========================================================
# Logging (terminal + file)
# =========================================================
RUN_BASE.mkdir(parents=True, exist_ok=True)
RUN_DIR = RUN_BASE / datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RUN_DIR / "training.log"


class Logger:
    def __init__(self, file_path: Path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_PATH)
print(f"📄 Logging attivo → {LOG_PATH}")
print("Using device:", DEVICE)


# =========================================================
# Dataset utilities
# =========================================================
def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class MultiLabelSeverityDataset(Dataset):
    def __init__(self, root: Path, split: str, classes: List[str], tf=None):
        self.root = root
        self.split = split
        self.classes = classes
        self.tf = tf
        self.rows = read_jsonl(root / split / "labels.jsonl")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        rel = Path(str(r["file"]).replace("\\", "/"))  # fix Windows → Linux
        img_path = self.root / rel
        img = Image.open(img_path).convert("RGB")

        if self.tf is not None:
            img = self.tf(img)

        y = torch.tensor([r["labels"][c] for c in self.classes], dtype=torch.float32)
        s = torch.tensor([r["severity"][c] for c in self.classes], dtype=torch.float32)
        return img, y, s


# =========================================================
# Model: backbone + two heads
# =========================================================
class MultiHeadClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_cls = nn.Linear(in_features, num_classes)  # logits
        self.head_sev = nn.Linear(in_features, num_classes)  # logits -> sigmoid -> [0,1]

    def forward(self, x):
        feat = self.backbone(x)
        cls_logits = self.head_cls(feat)
        sev_logits = self.head_sev(feat)
        return cls_logits, sev_logits


# =========================================================
# Metrics
# =========================================================
def f1_micro_macro(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    eps = 1e-9
    # micro
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1_micro = 2 * prec * rec / (prec + rec + eps)

    # macro
    f1s = []
    for c in range(y_true.shape[1]):
        tpc = (y_true[:, c] * y_pred[:, c]).sum()
        fpc = ((1 - y_true[:, c]) * y_pred[:, c]).sum()
        fnc = (y_true[:, c] * (1 - y_pred[:, c])).sum()
        pre = tpc / (tpc + fpc + eps)
        re = tpc / (tpc + fnc + eps)
        f1 = 2 * pre * re / (pre + re + eps)
        f1s.append(f1)
    f1_macro = float(np.mean(f1s))
    return float(f1_micro), float(f1_macro)


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, float]:
    eps = 1e-9
    out: Dict[str, float] = {}
    for i, c in enumerate(classes):
        tp = (y_true[:, i] * y_pred[:, i]).sum()
        fp = ((1 - y_true[:, i]) * y_pred[:, i]).sum()
        fn = (y_true[:, i] * (1 - y_pred[:, i])).sum()
        pre = tp / (tp + fp + eps)
        re = tp / (tp + fn + eps)
        f1 = 2 * pre * re / (pre + re + eps)
        out[c] = float(f1)
    return out


def severity_mae(y_true_lbl: np.ndarray, s_true: np.ndarray, s_pred: np.ndarray) -> float:
    # MAE only where label=1
    mask = (y_true_lbl > 0.5)
    if mask.sum() == 0:
        return float("nan")
    return float(np.abs(s_true[mask] - s_pred[mask]).mean())


# =========================================================
# Class imbalance helper: pos_weight for BCEWithLogitsLoss
# pos_weight[c] = (Nneg / Npos) for class c
# =========================================================
def compute_pos_weight(train_rows: List[dict], classes: List[str]) -> torch.Tensor:
    pos = np.zeros(len(classes), dtype=np.float64)
    neg = np.zeros(len(classes), dtype=np.float64)

    for r in train_rows:
        for i, c in enumerate(classes):
            if r["labels"][c] == 1:
                pos[i] += 1
            else:
                neg[i] += 1

    # avoid division by zero
    pos = np.maximum(pos, 1.0)
    w = neg / pos
    return torch.tensor(w, dtype=torch.float32)


# =========================================================
# Train / Eval loop
# =========================================================
@torch.no_grad()
def _concat_np(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    bce_loss: nn.Module,
    huber_loss: nn.Module,
    train: bool,
    classes: List[str],
):
    model.train(train)

    total_loss = 0.0
    total_cls = 0.0
    total_sev = 0.0

    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    all_s_true: List[np.ndarray] = []
    all_s_pred: List[np.ndarray] = []

    for x, y, s in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        s = s.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            cls_logits, sev_logits = model(x)

            # classification loss
            loss_cls = bce_loss(cls_logits, y)

            # severity prediction in [0,1]
            sev_pred = torch.sigmoid(sev_logits)

            # mask severity only where label=1
            mask = y > 0.5
            if mask.any():
                loss_sev = huber_loss(sev_pred[mask], s[mask])
            else:
                loss_sev = torch.zeros((), device=DEVICE)

            loss = loss_cls + LAMBDA_SEVERITY * loss_sev

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_cls += float(loss_cls.item())
        total_sev += float(loss_sev.item())

        # metrics
        probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        s_np = s.detach().cpu().numpy()
        s_pred_np = sev_pred.detach().cpu().numpy()

        all_p.append(probs)
        all_y.append(y_np)
        all_s_true.append(s_np)
        all_s_pred.append(s_pred_np)

    all_p = _concat_np(all_p)
    all_y = _concat_np(all_y)
    all_s_true = _concat_np(all_s_true)
    all_s_pred = _concat_np(all_s_pred)

    y_hat = (all_p >= THRESH).astype(np.float32)
    f1_micro, f1_macro = f1_micro_macro(all_y, y_hat)
    f1_by_class = per_class_f1(all_y, y_hat, classes)
    sev_mae = severity_mae(all_y, all_s_true, all_s_pred)

    n_batches = max(1, len(loader))
    return {
        "loss": total_loss / n_batches,
        "loss_cls": total_cls / n_batches,
        "loss_sev": total_sev / n_batches,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "sev_mae": sev_mae,
        "f1_by_class": f1_by_class,
    }


# =========================================================
# Plot helper
# =========================================================
def plot_curve(y1, y2, title, ylabel, name):
    plt.figure()
    plt.plot(y1, label="train")
    plt.plot(y2, label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / name)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    # read classes
    classes = json.loads((DATASET_ROOT / "meta" / "classes.json").read_text(encoding="utf-8"))
    num_classes = len(classes)
    print("Classes:", classes)
    print("Num classes:", num_classes)

    # ---- transforms ----
    # NOTE: images are already 256x384. Keep augmentations conservative.
    train_tf = transforms.Compose([
        transforms.Resize((256, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256, 384)),
        transforms.ToTensor(),
    ])

    # ---- datasets ----
    train_ds = MultiLabelSeverityDataset(DATASET_ROOT, "train", classes, tf=train_tf)
    val_ds = MultiLabelSeverityDataset(DATASET_ROOT, "val", classes, tf=eval_tf)
    test_ds = MultiLabelSeverityDataset(DATASET_ROOT, "test", classes, tf=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    print(f"Dataset sizes: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    # ---- model ----
    model = MultiHeadClassifier(num_classes=num_classes).to(DEVICE)

    # ---- losses ----
    # compute pos_weight from TRAIN split to handle class imbalance
    train_rows = read_jsonl(DATASET_ROOT / "train" / "labels.jsonl")
    pos_weight = compute_pos_weight(train_rows, classes).to(DEVICE)
    print("pos_weight:", pos_weight.detach().cpu().numpy().tolist())

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    huber_loss = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---- training ----
    best_score = -1.0
    patience = 0
    best_path = RUN_DIR / "best_model.pt"

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1micro": [], "val_f1micro": [],
        "train_f1macro": [], "val_f1macro": [],
        "train_sev_mae": [], "val_sev_mae": [],
    }

    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{NUM_EPOCHS} =====")
        e0 = time.time()

        tr = run_epoch(model, train_loader, optimizer, bce_loss, huber_loss, train=True, classes=classes)
        va = run_epoch(model, val_loader, None, bce_loss, huber_loss, train=False, classes=classes)

        dt = time.time() - e0
        print(
            f"Train loss={tr['loss']:.4f} (cls={tr['loss_cls']:.4f}, sev={tr['loss_sev']:.4f}) | "
            f"F1micro={tr['f1_micro']:.4f} F1macro={tr['f1_macro']:.4f} | sevMAE={tr['sev_mae']:.4f}"
        )
        print(
            f"Val   loss={va['loss']:.4f} (cls={va['loss_cls']:.4f}, sev={va['loss_sev']:.4f}) | "
            f"F1micro={va['f1_micro']:.4f} F1macro={va['f1_macro']:.4f} | sevMAE={va['sev_mae']:.4f}"
        )
        print(f"⏱️  epoch time: {dt:.1f}s")

        # save per-class report on VAL each epoch (useful for debugging)
        (RUN_DIR / "per_class_f1_val.json").write_text(json.dumps(va["f1_by_class"], indent=2), encoding="utf-8")

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_f1micro"].append(tr["f1_micro"])
        history["val_f1micro"].append(va["f1_micro"])
        history["train_f1macro"].append(tr["f1_macro"])
        history["val_f1macro"].append(va["f1_macro"])
        history["train_sev_mae"].append(tr["sev_mae"])
        history["val_sev_mae"].append(va["sev_mae"])

        # Best criterion: VAL F1 micro
        score = va["f1_micro"]
        if score > best_score:
            best_score = score
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "thresh": THRESH,
                    "pos_weight": pos_weight.detach().cpu().numpy().tolist(),
                    "epoch": epoch,
                    "val_f1_micro": best_score,
                },
                best_path,
            )
            print(f"💾 Best model saved → {best_path} (best VAL F1micro={best_score:.4f})")
        else:
            patience += 1
            print(f"⏳ Early stopping counter: {patience}/{PATIENCE}")
            if patience >= PATIENCE:
                print("\n🛑 EARLY STOPPING (based on VAL)")
                break

    total = time.time() - t0
    print(f"\n⏱️ Total training time: {total/60:.1f} min")

    # ---- save curves ----
    (RUN_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_curve(history["train_loss"], history["val_loss"], "Loss", "loss", "loss_curve.png")
    plot_curve(history["train_f1micro"], history["val_f1micro"], "F1 micro", "F1", "f1_micro.png")
    plot_curve(history["train_f1macro"], history["val_f1macro"], "F1 macro", "F1", "f1_macro.png")
    plot_curve(history["train_sev_mae"], history["val_sev_mae"], "Severity MAE", "MAE", "sev_mae.png")

    # =========================================================
    # FINAL TEST (evaluate once, after selecting best model)
    # =========================================================
    print("\n===== FINAL TEST (best model) =====")
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    te = run_epoch(model, test_loader, None, bce_loss, huber_loss, train=False, classes=classes)
    print(
        f"Test  loss={te['loss']:.4f} (cls={te['loss_cls']:.4f}, sev={te['loss_sev']:.4f}) | "
        f"F1micro={te['f1_micro']:.4f} F1macro={te['f1_macro']:.4f} | sevMAE={te['sev_mae']:.4f}"
    )

    (RUN_DIR / "per_class_f1_test.json").write_text(json.dumps(te["f1_by_class"], indent=2), encoding="utf-8")

    summary = {
        "best_model_path": str(best_path),
        "best_val_f1_micro": float(ckpt.get("val_f1_micro", -1.0)),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "test": {
            "loss": te["loss"],
            "loss_cls": te["loss_cls"],
            "loss_sev": te["loss_sev"],
            "f1_micro": te["f1_micro"],
            "f1_macro": te["f1_macro"],
            "sev_mae": te["sev_mae"],
        },
        "thresh": THRESH,
        "lambda_severity": LAMBDA_SEVERITY,
        "pos_weight": ckpt.get("pos_weight", None),
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n📂 Tutti i risultati salvati in: {RUN_DIR}")
    print(f"📄 Log completo: {LOG_PATH}")
    print(f"✅ Best model: {best_path}")


if __name__ == "__main__":
    main()
