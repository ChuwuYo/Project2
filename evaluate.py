"""评估脚本：加载训练好的模型，在指定子集上计算准确率与混淆矩阵。

本项目中只有这个脚本会读取 PrivateTest。评估过程没有反向传播，也不调整任何参数；
是否使用 TTA 由训练时在 PublicTest 上得出的 summary.json 决定。

用法：
    python evaluate.py                       # 用 final_runs.txt 中的最终模型做 PrivateTest 评估
    python evaluate.py --runs runs/A3_s0                      # 单模型，PrivateTest 最终评估
    python evaluate.py --runs runs/A3_s0 runs/A3_s1 runs/A3_s2   # 多模型概率平均（集成）
    python evaluate.py --runs runs/A3_s0 --split PublicTest   # 在验证集上比较方案
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from dataset import CLASSES, load_split
from utils import accuracy, final_runs, get_device, load_runs, make_loader, plot_confusion, predict_probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FER2013 模型评估")
    p.add_argument("--runs", type=Path, nargs="+", default=None,
                   help="一个或多个训练输出目录；缺省时读取 final_runs.txt")
    p.add_argument("--split", choices=["PublicTest", "PrivateTest"], default="PrivateTest")
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()
    args.runs = args.runs or final_runs()
    return args


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"设备: {device}  |  子集: {args.split}  |  模型数: {len(args.runs)}")
    images, labels = load_split(args.split)
    loader = make_loader(images, labels, train=False, batch_size=256)
    print(f"已加载 {len(labels)} 张图片，正在加载模型 ...")

    # 每个模型按自己的 summary.json 决定是否 TTA，再把各模型的概率取平均
    probs, members = [], []
    n = len(args.runs)
    for i, (run, (model, use_tta, ckpt)) in enumerate(
            zip(args.runs, load_runs(args.runs, device), strict=True), start=1):
        p, y = predict_probs(model, loader, device, tta=use_tta, desc=f"[{i}/{n}] {run.name}")
        probs.append(p)
        members.append({"run": str(run), "best_epoch": ckpt["epoch"], "tta": use_tta,
                        "acc": round(accuracy(p, y), 4)})
        print(f"[{i}/{n}] {run.name:<8} 单模型准确率 {members[-1]['acc']:.4f}  (TTA={use_tta})")
    mean_probs = sum(probs) / len(probs)
    acc = accuracy(mean_probs, y)
    pred = mean_probs.argmax(dim=1).numpy()

    print("正在生成报告和混淆矩阵 ...\n")

    # ---- 保存结果 ----
    name = args.split if len(args.runs) == 1 else f"{args.split}_ensemble{len(args.runs)}"
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y.numpy(), pred, labels=range(len(CLASSES)))
    report = classification_report(y.numpy(), pred, target_names=CLASSES, digits=4,
                                   zero_division=0)
    recall = (np.diag(cm) / cm.sum(axis=1)).round(4)
    metrics = {"split": args.split, "n": len(y), "accuracy": round(acc, 4), "members": members,
               "recall": dict(zip(CLASSES, recall.tolist(), strict=True)), "confusion_matrix": cm.tolist()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out_dir / "classification_report.txt").write_text(report)
    plot_confusion(cm, out_dir / "confusion_matrix.png", f"{args.split}  acc = {acc:.4f}")

    print(report)
    print(f"{args.split} 准确率: {acc:.4f}  ({len(y)} 张)  结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
