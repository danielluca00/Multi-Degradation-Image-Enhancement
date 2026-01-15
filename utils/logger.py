# utils/logger.py
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class RunPaths:
    run_dir: str
    train_csv: str
    train_jsonl: str
    test_csv: str
    test_jsonl: str
    summary_json: str
    config_copy: str


class ExperimentLogger:
    """
    Writes:
      - train.csv / train.jsonl
      - test.csv / test.jsonl
      - summary.json
      - optional config.json copy
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = (config.get("logging", {}) or {})
        self.enabled = bool(self.cfg.get("enabled", False))

        self.run_paths: Optional[RunPaths] = None

        self._train_csv_writer: Optional[csv.DictWriter] = None
        self._test_csv_writer: Optional[csv.DictWriter] = None
        self._train_csv_f = None
        self._test_csv_f = None

        self._train_fieldnames = None
        self._test_fieldnames = None

        self._summary: Dict[str, Any] = {}

        if not self.enabled:
            return

        task_name = str(config.get("name", "run"))
        root_dir = str(self.cfg.get("root_dir", "runs"))
        run_dir = os.path.join(root_dir, task_name, _now_stamp())
        _ensure_dir(run_dir)

        self.run_paths = RunPaths(
            run_dir=run_dir,
            train_csv=os.path.join(run_dir, "train.csv"),
            train_jsonl=os.path.join(run_dir, "train.jsonl"),
            test_csv=os.path.join(run_dir, "test.csv"),
            test_jsonl=os.path.join(run_dir, "test.jsonl"),
            summary_json=os.path.join(run_dir, "summary.json"),
            config_copy=os.path.join(run_dir, "config.json"),
        )

        if bool(self.cfg.get("save_config_copy", True)):
            with open(self.run_paths.config_copy, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        self._summary = {
            "task": task_name,
            "created_at": datetime.now().isoformat(),
            "run_dir": run_dir,
        }
        self._write_summary()

    def run_dir(self) -> Optional[str]:
        return self.run_paths.run_dir if self.run_paths else None

    def _append_jsonl(self, path: str, row: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _open_csv_if_needed(self, kind: str, fieldnames) -> None:
        assert self.run_paths is not None

        if kind == "train":
            if self._train_csv_f is None:
                self._train_csv_f = open(self.run_paths.train_csv, "a", newline="", encoding="utf-8")
                self._train_csv_writer = csv.DictWriter(self._train_csv_f, fieldnames=fieldnames)
                if self._train_csv_f.tell() == 0:
                    self._train_csv_writer.writeheader()
            return

        if kind == "test":
            if self._test_csv_f is None:
                self._test_csv_f = open(self.run_paths.test_csv, "a", newline="", encoding="utf-8")
                self._test_csv_writer = csv.DictWriter(self._test_csv_f, fieldnames=fieldnames)
                if self._test_csv_f.tell() == 0:
                    self._test_csv_writer.writeheader()
            return

        raise ValueError(f"Unknown CSV kind: {kind}")

    def _csv_write_row(self, kind: str, row: Dict[str, Any]) -> None:
        assert self.run_paths is not None

        if kind == "train":
            if self._train_fieldnames is None:
                self._train_fieldnames = list(row.keys())
                self._open_csv_if_needed("train", self._train_fieldnames)
            self._train_csv_writer.writerow(row)
            self._train_csv_f.flush()
            return

        if kind == "test":
            if self._test_fieldnames is None:
                self._test_fieldnames = list(row.keys())
                self._open_csv_if_needed("test", self._test_fieldnames)
            self._test_csv_writer.writerow(row)
            self._test_csv_f.flush()
            return

        raise ValueError(f"Unknown CSV kind: {kind}")

    def log_train(self, row: Dict[str, Any]) -> None:
        if not self.enabled or self.run_paths is None:
            return

        tr_cfg = (self.cfg.get("train", {}) or {})
        if bool(tr_cfg.get("save_jsonl", True)):
            self._append_jsonl(self.run_paths.train_jsonl, row)
        if bool(tr_cfg.get("save_csv", True)):
            self._csv_write_row("train", row)

    def log_test(self, row: Dict[str, Any]) -> None:
        if not self.enabled or self.run_paths is None:
            return

        te_cfg = (self.cfg.get("test", {}) or {})
        if bool(te_cfg.get("save_jsonl", True)):
            self._append_jsonl(self.run_paths.test_jsonl, row)
        if bool(te_cfg.get("save_csv", True)):
            self._csv_write_row("test", row)

    def set_summary(self, summary: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._summary.update(summary)
        self._write_summary()

    def _write_summary(self) -> None:
        if not self.enabled or self.run_paths is None:
            return
        with open(self.run_paths.summary_json, "w", encoding="utf-8") as f:
            json.dump(self._summary, f, indent=2, ensure_ascii=False)

    def close(self) -> None:
        if self._train_csv_f:
            self._train_csv_f.close()
        if self._test_csv_f:
            self._test_csv_f.close()
