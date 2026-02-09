# classification/train_multilabel_classifier.py
from __future__ import annotations

import argparse
import json
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
# CONFIG (defaults; can be overridden via CLI)
# =========================================================
DATASET_ROOT = Path("classifier_dataset")
RUN_BASE = Path("runs_classifier")

BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4
PATIENCE = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# default threshold (used if you don't tune)
DEFAULT_THRESH = 0.5

# loss weights
LAMBDA_SEVERITY = 0.5  # weight of severity loss

# dataloader
NUM_WORKERS = 4
PIN_MEMORY = True

# threshold tuning
THRESH_GRID = [float(x) for x in np.linspace(0.05, 0.95, 19)]  # 0.05..0.95 step 0.05


# =========================================================
# Logging (terminal + file)
# =========================================================
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
    # y_* shape [N,C] binary
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
# Helpers: run model to collect probs + y + severities
# =========================================================
@torch.no_grad()
def collect_outputs(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      probs_cls: [N,C] sigmoid(cls_logits)
      y_true:   [N,C] 0/1
      s_true:   [N,C] severity ground truth
      s_pred:   [N,C] sigmoid(sev_logits)
    """
    model.eval()

    all_p: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_s_true: List[np.ndarray] = []
    all_s_pred: List[np.ndarray] = []

    for x, y, s in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        s = s.to(DEVICE, non_blocking=True)

        cls_logits, sev_logits = model(x)
        probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
        sev_pred = torch.sigmoid(sev_logits).detach().cpu().numpy()

        all_p.append(probs)
        all_y.append(y.detach().cpu().numpy())
        all_s_true.append(s.detach().cpu().numpy())
        all_s_pred.append(sev_pred)

    probs_cls = np.concatenate(all_p, axis=0) if all_p else np.zeros((0, 0), dtype=np.float32)
    y_true = np.concatenate(all_y, axis=0) if all_y else np.zeros((0, 0), dtype=np.float32)
    s_true = np.concatenate(all_s_true, axis=0) if all_s_true else np.zeros((0, 0), dtype=np.float32)
    s_pred = np.concatenate(all_s_pred, axis=0) if all_s_pred else np.zeros((0, 0), dtype=np.float32)
    return probs_cls, y_true, s_true, s_pred


def apply_thresholds(probs: np.ndarray, thresholds: List[float]) -> np.ndarray:
    th = np.array(thresholds, dtype=np.float32).reshape(1, -1)
    return (probs >= th).astype(np.float32)


def tune_thresholds_per_class_for_f1(
    probs: np.ndarray,
    y_true: np.ndarray,
    classes: List[str],
    grid: List[float],
) -> Dict:
    """
    Finds per-class threshold that maximizes that class F1 on validation.
    Also reports micro/macro using those thresholds.
    """
    C = y_true.shape[1]
    best_thr = [DEFAULT_THRESH] * C
    best_f1 = [0.0] * C

    # per-class
    for ci in range(C):
        yt = y_true[:, ci]
        if yt.sum() == 0:
            # no positives in val; keep default threshold
            best_thr[ci] = DEFAULT_THRESH
            best_f1[ci] = 0.0
            continue

        best_ci_f1 = -1.0
        best_ci_thr = DEFAULT_THRESH
        for t in grid:
            yp = (probs[:, ci] >= t).astype(np.float32)
            # compute f1 for this class
            eps = 1e-9
            tp = (yt * yp).sum()
            fp = ((1 - yt) * yp).sum()
            fn = (yt * (1 - yp)).sum()
            pre = tp / (tp + fp + eps)
            re = tp / (tp + fn + eps)
            f1 = 2 * pre * re / (pre + re + eps)
            if f1 > best_ci_f1:
                best_ci_f1 = float(f1)
                best_ci_thr = float(t)

        best_thr[ci] = best_ci_thr
        best_f1[ci] = best_ci_f1

    # overall metrics using per-class thresholds
    y_hat = apply_thresholds(probs, best_thr)
    f1_micro, f1_macro = f1_micro_macro(y_true, y_hat)
    f1_by_class = per_class_f1(y_true, y_hat, classes)

    report = {
        "objective": "maximize per-class F1 on VAL (grid search), then evaluate overall",
        "grid": grid,
        "thresholds": {c: float(best_thr[i]) for i, c in enumerate(classes)},
        "best_class_f1_on_val": {c: float(best_f1[i]) for i, c in enumerate(classes)},
        "val_f1_micro": float(f1_micro),
        "val_f1_macro": float(f1_macro),
        "val_f1_by_class": f1_by_class,
    }
    return report


# =========================================================
# Train / Eval loop (uses thresholds passed in)
# =========================================================
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    bce_loss: nn.Module,
    huber_loss: nn.Module,
    train: bool,
    classes: List[str],
    thresholds: List[float],
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

    all_p = np.concatenate(all_p, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_s_true = np.concatenate(all_s_true, axis=0)
    all_s_pred = np.concatenate(all_s_pred, axis=0)

    y_hat = apply_thresholds(all_p, thresholds)

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
def plot_curve(run_dir: Path, y1, y2, title, ylabel, name):
    plt.figure()
    plt.plot(y1, label="train")
    plt.plot(y2, label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / name)
    plt.close()


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default=str(DATASET_ROOT))
    p.add_argument("--run_dir", type=str, default="", help="If empty -> create a new run dir. If set -> reuse it.")
    p.add_argument("--checkpoint", type=str, default="", help="If set, load this checkpoint for tune/test. If empty, use run_dir/best_model.pt.")
    p.add_argument("--train", action="store_true", help="Run training loop.")
    p.add_argument("--tune_thresh", action="store_true", help="Tune per-class thresholds on VAL after training (or from checkpoint).")
    p.add_argument("--test", action="store_true", help="Run FINAL TEST (after optionally tuning thresholds).")

    # threshold tuning grid
    p.add_argument("--th_min", type=float, default=0.05)
    p.add_argument("--th_max", type=float, default=0.95)
    p.add_argument("--th_steps", type=int, default=19)

    # override hyperparams (optional)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    return p.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    # Run dir handling
    RUN_BASE.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = RUN_BASE / datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "training.log"
    sys.stdout = Logger(log_path)
    print(f"📄 Logging attivo → {log_path}")
    print("Using device:", DEVICE)
    print("Dataset root:", dataset_root.resolve())
    print("Run dir:", run_dir.resolve())

    # If user runs script without flags, do the sensible default:
    # train + tune + test
    if not (args.train or args.tune_thresh or args.test):
        args.train = True
        args.tune_thresh = True
        args.test = True

    # Read classes
    classes = json.loads((dataset_root / "meta" / "classes.json").read_text(encoding="utf-8"))
    num_classes = len(classes)
    print("Classes:", classes)
    print("Num classes:", num_classes)

    # Transforms
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

    # Datasets + loaders
    train_ds = MultiLabelSeverityDataset(dataset_root, "train", classes, tf=train_tf)
    val_ds = MultiLabelSeverityDataset(dataset_root, "val", classes, tf=eval_tf)
    test_ds = MultiLabelSeverityDataset(dataset_root, "test", classes, tf=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=PIN_MEMORY
    )
    print(f"Dataset sizes: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    # Model
    model = MultiHeadClassifier(num_classes=num_classes).to(DEVICE)

    # Losses
    train_rows = read_jsonl(dataset_root / "train" / "labels.jsonl")
    pos_weight = compute_pos_weight(train_rows, classes).to(DEVICE)
    print("pos_weight:", pos_weight.detach().cpu().numpy().tolist())

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    huber_loss = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Thresholds used during training/val metrics
    thresholds = [DEFAULT_THRESH] * num_classes

    best_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.json"

    # -------------------------
    # TRAIN
    # -------------------------
    if args.train:
        best_score = -1.0
        patience = 0

        history = {
            "train_loss": [], "val_loss": [],
            "train_f1micro": [], "val_f1micro": [],
            "train_f1macro": [], "val_f1macro": [],
            "train_sev_mae": [], "val_sev_mae": [],
        }

        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            print(f"\n===== EPOCH {epoch}/{args.epochs} =====")
            e0 = time.time()

            tr = run_epoch(model, train_loader, optimizer, bce_loss, huber_loss, train=True, classes=classes, thresholds=thresholds)
            va = run_epoch(model, val_loader, None, bce_loss, huber_loss, train=False, classes=classes, thresholds=thresholds)

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

            # save per-class report on VAL each epoch
            (run_dir / "per_class_f1_val.json").write_text(json.dumps(va["f1_by_class"], indent=2), encoding="utf-8")

            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(va["loss"])
            history["train_f1micro"].append(tr["f1_micro"])
            history["val_f1micro"].append(va["f1_micro"])
            history["train_f1macro"].append(tr["f1_macro"])
            history["val_f1macro"].append(va["f1_macro"])
            history["train_sev_mae"].append(tr["sev_mae"])
            history["val_sev_mae"].append(va["sev_mae"])

            # Best criterion: VAL F1 micro (with default thresholds)
            score = va["f1_micro"]
            if score > best_score:
                best_score = score
                patience = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": classes,
                        "default_thresh": DEFAULT_THRESH,
                        "pos_weight": pos_weight.detach().cpu().numpy().tolist(),
                        "epoch": epoch,
                        "val_f1_micro": best_score,
                    },
                    best_path,
                )
                print(f"💾 Best model saved → {best_path} (best VAL F1micro={best_score:.4f})")
            else:
                patience += 1
                print(f"⏳ Early stopping counter: {patience}/{args.patience}")
                if patience >= args.patience:
                    print("\n🛑 EARLY STOPPING (based on VAL)")
                    break

        total = time.time() - t0
        print(f"\n⏱️ Total training time: {total/60:.1f} min")

        # Save history + curves
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_curve(run_dir, history["train_loss"], history["val_loss"], "Loss", "loss", "loss_curve.png")
        plot_curve(run_dir, history["train_f1micro"], history["val_f1micro"], "F1 micro", "F1", "f1_micro.png")
        plot_curve(run_dir, history["train_f1macro"], history["val_f1macro"], "F1 macro", "F1", "f1_macro.png")
        plot_curve(run_dir, history["train_sev_mae"], history["val_sev_mae"], "Severity MAE", "MAE", "sev_mae.png")

    # -------------------------
    # LOAD CHECKPOINT for tune/test
    # -------------------------
    ckpt_path = Path(args.checkpoint) if args.checkpoint else best_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("\n===== LOADING CHECKPOINT =====")
    print("Checkpoint:", ckpt_path.resolve())
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -------------------------
    # TUNE THRESHOLDS on VAL
    # -------------------------
    tuned_thresholds: List[float] = thresholds[:]  # fallback
    tune_report = None

    if args.tune_thresh:
        print("\n===== THRESHOLD TUNING (VAL) =====")
        grid = [float(x) for x in np.linspace(args.th_min, args.th_max, args.th_steps)]
        probs_val, y_val, s_val, s_pred_val = collect_outputs(model, val_loader)

        tune_report = tune_thresholds_per_class_for_f1(
            probs=probs_val,
            y_true=y_val,
            classes=classes,
            grid=grid,
        )

        # extract per-class thresholds in correct order
        tuned_thresholds = [tune_report["thresholds"][c] for c in classes]

        (run_dir / "thresholds_val.json").write_text(json.dumps(tune_report, indent=2), encoding="utf-8")
        print("✅ Saved:", (run_dir / "thresholds_val.json").resolve())
        print("Tuned thresholds:", {c: float(tuned_thresholds[i]) for i, c in enumerate(classes)})
        print(f"VAL F1micro (tuned): {tune_report['val_f1_micro']:.4f} | VAL F1macro (tuned): {tune_report['val_f1_macro']:.4f}")

    # -------------------------
    # FINAL TEST
    # -------------------------
    if args.test:
        print("\n===== FINAL TEST =====")
        te = run_epoch(model, test_loader, None, bce_loss, huber_loss, train=False, classes=classes, thresholds=tuned_thresholds)
        print(
            f"Test  loss={te['loss']:.4f} (cls={te['loss_cls']:.4f}, sev={te['loss_sev']:.4f}) | "
            f"F1micro={te['f1_micro']:.4f} F1macro={te['f1_macro']:.4f} | sevMAE={te['sev_mae']:.4f}"
        )
        (run_dir / "per_class_f1_test.json").write_text(json.dumps(te["f1_by_class"], indent=2), encoding="utf-8")

        summary = {
            "run_dir": str(run_dir),
            "dataset_root": str(dataset_root),
            "checkpoint_used": str(ckpt_path),
            "device": DEVICE,
            "classes": classes,
            "default_threshold": DEFAULT_THRESH,
            "tuned_thresholds_used": {c: float(tuned_thresholds[i]) for i, c in enumerate(classes)},
            "lambda_severity": LAMBDA_SEVERITY,
            "pos_weight": ckpt.get("pos_weight", None),
            "best_val_f1_micro_default_thresh": float(ckpt.get("val_f1_micro", -1.0)),
            "best_epoch": int(ckpt.get("epoch", -1)),
            "test": {
                "loss": te["loss"],
                "loss_cls": te["loss_cls"],
                "loss_sev": te["loss_sev"],
                "f1_micro": te["f1_micro"],
                "f1_macro": te["f1_macro"],
                "sev_mae": te["sev_mae"],
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"\n📂 Tutti i risultati salvati in: {run_dir}")
        print(f"📄 Log completo: {log_path}")
        print(f"✅ Checkpoint usato: {ckpt_path}")

    print("\n[OK]")


if __name__ == "__main__":
    main()
