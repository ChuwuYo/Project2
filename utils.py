"""训练、评估、前端共用的工具函数。"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境（Colab / 服务器）也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLASSES, FERDataset, build_transform
from model import build_model


def set_seed(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 的随机种子，使实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_workers() -> int:
    """Windows 上多进程加载容易出问题，默认 0；Linux（Colab）默认 2。"""
    return 0 if os.name == "nt" else 2


def make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    train: bool,
    batch_size: int,
    augment: bool = False,
    random_erasing: bool = False,
    workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    """训练集打乱并（可选）增强；验证/测试集固定顺序、只做标准预处理。"""
    transform = build_transform(train=train and augment, random_erasing=random_erasing)
    return DataLoader(
        FERDataset(images, labels, transform),
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed) if train else None,
    )


@torch.inference_mode()
def predict_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                  tta: bool = False, desc: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """对整个数据集预测，返回 (softmax 概率 (N, 7), 真实标签 (N,))。给 desc 时显示进度条。"""
    model.eval()
    all_probs, all_labels = [], []
    batches = tqdm(loader, desc=desc, unit=" batch", leave=False) if desc else loader
    for x, y in batches:
        all_probs.append(model_probs(model, x.to(device), tta).cpu())
        all_labels.append(y)
    return torch.cat(all_probs), torch.cat(all_labels)


@torch.inference_mode()
def model_probs(model: torch.nn.Module, x: torch.Tensor, tta: bool) -> torch.Tensor:
    """一个 batch 的 softmax 概率；评估和前端都调用它，保证推理方式一致。"""
    probs = model(x).softmax(dim=1)
    if tta:  # 水平翻转后再预测一次，两次概率取平均
        probs = (probs + model(torch.flip(x, dims=[3])).softmax(dim=1)) / 2
    return probs.float()


def accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return (probs.argmax(dim=1) == labels).float().mean().item()


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """从 checkpoint 恢复模型（eval 模式），同时返回 checkpoint 中保存的信息。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(dropout=ckpt["config"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


FINAL_RUNS_FILE = Path(__file__).resolve().parent / "runs" / "final_runs.txt"
BEST_SINGLE_FILE = Path(__file__).resolve().parent / "runs" / "best_single.txt"


def final_runs(single: bool = False) -> list[Path]:
    """读取最终模型目录（都按 PublicTest 选定，每行一个，# 开头为注释）。

    single=False 读 runs/final_runs.txt（多模型概率平均）；single=True 读 runs/best_single.txt（最佳单模型）。
    """
    path = BEST_SINGLE_FILE if single else FINAL_RUNS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    return [path.parent / ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def load_runs(runs: list[Path], device: torch.device) -> list[tuple[torch.nn.Module, bool, dict]]:
    """加载一个或多个训练目录的 best.pt；是否 TTA 读取该目录 summary.json（PublicTest 上的决定）。"""
    loaded = []
    for run in runs:
        model, ckpt = load_model(Path(run) / "best.pt", device)
        use_tta = json.loads((Path(run) / "summary.json").read_text())["use_tta"]
        loaded.append((model, use_tta, ckpt))
    return loaded


def append_csv(path: Path, row: dict) -> None:
    """向 CSV 追加一行；文件不存在时先写表头。每轮训练后立即写盘。"""
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def plot_curves(log_csv: Path, out_png: Path) -> None:
    """根据 log.csv 画 loss 与 accuracy 曲线。"""
    with open(log_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epoch = [int(r["epoch"]) for r in rows]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (("train", "Training"), ("val", "PublicTest")):
        ax_loss.plot(epoch, [float(r[f"{key}_loss"]) for r in rows], label=label)
        ax_acc.plot(epoch, [float(r[f"{key}_acc"]) for r in rows], label=label)
    for ax, title in ((ax_loss, "Loss"), (ax_acc, "Accuracy")):
        ax.set(title=title, xlabel="epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_png: Path, title: str) -> None:
    """画按行归一化的混淆矩阵（每行 = 一个真实类别的召回分布），格子里同时标数量。"""
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
