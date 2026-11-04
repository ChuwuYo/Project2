"""训练脚本：只用 Training 更新参数，只用 PublicTest 做早停、选 checkpoint 和 TTA 决策。

用法（自己训练时请加 --out runs_local，避免和 runs/ 里的最终模型混在一起）：
    python train.py --exp A1 --seed 0 --out runs_local            # 正式训练
    python train.py --exp A1 --seed 0 --out runs_local --resume   # 中断后从 last.pt 继续
    python train.py --exp A1 --subset 2000 --epochs 2 --out runs_local   # 小样本试跑

如果输出目录里已经有 best.pt（例如 runs/A1_s0 是最终模型之一），不加 --resume 会直接报错退出，
不会覆盖已有结果。

输出目录 <out>/<exp>_s<seed>/（默认 out 为 runs）：
    config.json   本次训练的全部配置
    log.csv       每轮 train/val 的 loss、acc、学习率、耗时
    best.pt       PublicTest 准确率最高的模型
    last.pt       最新一轮的完整状态（用于断点续训）
    curves.png    loss / accuracy 曲线
    summary.json  最佳轮次、PublicTest 准确率、是否启用 TTA
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import load_split
from model import build_model
from utils import (accuracy, append_csv, default_workers, get_device, load_model, make_loader,
                   plot_curves, predict_probs, set_seed)

# 消融实验：每一组只比上一组多一个改动，其余条件完全相同。
EXPERIMENTS = {
    "A0": dict(augment=False, label_smoothing=0.0, dropout=0.0, random_erasing=False),
    "A1": dict(augment=True, label_smoothing=0.0, dropout=0.0, random_erasing=False),
    "A2": dict(augment=True, label_smoothing=0.1, dropout=0.0, random_erasing=False),
    "A3": dict(augment=True, label_smoothing=0.1, dropout=0.4, random_erasing=True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FER2013 表情识别训练")
    p.add_argument("--exp", choices=EXPERIMENTS, default="A3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20, help="PublicTest 准确率连续多少轮不升就早停")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--subset", type=int, default=0, help="只取前 N 张训练/验证图，调试用")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--resume", action="store_true", help="从 last.pt 继续训练")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> tuple[float, float]:
    """训练一轮，返回 (平均 loss, 准确率)。注意：开启增强时这是在扰动后的图像上算的。"""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("loss 出现 NaN/Inf，停止训练")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.inference_mode()
def validate(model, loader, criterion, device) -> tuple[float, float]:
    """在 PublicTest 上计算 (平均 loss, 准确率)，不做任何增强。"""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def main() -> None:
    args = parse_args()
    config = {**vars(args), **EXPERIMENTS[args.exp], "out": str(args.out)}
    run_dir = args.out / f"{args.exp}_s{args.seed}"
    # 防止误覆盖：目录里已有训练好的模型，只有在能续训（有 last.pt 且加了 --resume）时才继续
    if (run_dir / "best.pt").exists() and not (args.resume and (run_dir / "last.pt").exists()):
        raise SystemExit(f"{run_dir} 里已经有训练好的模型（best.pt），为防止覆盖已停止。\n"
                         f"重新训练请换一个输出目录，例如加 --out runs_local；"
                         f"中断后接着训练请加 --resume（需要 last.pt）。")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print(f"设备: {device} | 实验: {args.exp} {EXPERIMENTS[args.exp]} | 输出: {run_dir}")

    # ---- 数据：只读取 Training 和 PublicTest ----
    x_train, y_train = load_split("Training")
    x_val, y_val = load_split("PublicTest")
    if args.subset:
        x_train, y_train = x_train[:args.subset], y_train[:args.subset]
        x_val, y_val = x_val[:args.subset], y_val[:args.subset]
    train_loader = make_loader(x_train, y_train, train=True, batch_size=args.batch_size,
                               augment=config["augment"], random_erasing=config["random_erasing"],
                               workers=args.workers, seed=args.seed)
    val_loader = make_loader(x_val, y_val, train=False, batch_size=256, workers=args.workers)
    print(f"Training {len(y_train)} 张 | PublicTest {len(y_val)} 张")

    # ---- 模型、损失、优化器、学习率调度 ----
    model = build_model(dropout=config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")  # GPU 上混合精度

    start_epoch, best_acc, best_epoch, bad_epochs = 1, 0.0, 0, 0
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "log.csv"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        for obj, key in ((model, "model"), (optimizer, "optimizer"),
                         (scheduler, "scheduler"), (scaler, "scaler")):
            obj.load_state_dict(state[key])
        start_epoch = state["epoch"] + 1
        best_acc, best_epoch, bad_epochs = state["best_acc"], state["best_epoch"], state["bad_epochs"]
        print(f"从第 {state['epoch']} 轮恢复，当前最佳 PublicTest acc = {best_acc:.4f}")
    else:
        log_path.unlink(missing_ok=True)  # 重新训练时清空旧日志
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    # ---- 训练循环 ----
    for epoch in range(start_epoch, args.epochs + 1):
        if bad_epochs >= args.patience:
            break
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        append_csv(log_path, dict(epoch=epoch, train_loss=round(train_loss, 4),
                                  train_acc=round(train_acc, 4), val_loss=round(val_loss, 4),
                                  val_acc=round(val_acc, 4), lr=lr, seconds=round(time.time() - t0, 1)))
        if val_acc > best_acc:
            best_acc, best_epoch, bad_epochs = val_acc, epoch, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "val_acc": val_acc}, best_path)
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_acc": best_acc, "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs}, last_path)
        print(f"[{epoch:3d}] train {train_loss:.4f}/{train_acc:.4f} | "
              f"val {val_loss:.4f}/{val_acc:.4f} | lr {lr:.5f} | best {best_acc:.4f}@{best_epoch} | "
              f"{time.time() - t0:.0f}s", flush=True)

    # ---- 收尾：用 PublicTest 决定是否启用翻转 TTA ----
    best_model, _ = load_model(best_path, device)
    acc_plain = accuracy(*predict_probs(best_model, val_loader, device, tta=False))
    acc_tta = accuracy(*predict_probs(best_model, val_loader, device, tta=True))
    summary = dict(exp=args.exp, seed=args.seed, best_epoch=best_epoch, val_acc=round(acc_plain, 4),
                   val_acc_tta=round(acc_tta, 4), use_tta=acc_tta > acc_plain)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_curves(log_path, run_dir / "curves.png")
    print("完成:", summary)


if __name__ == "__main__":
    main()
