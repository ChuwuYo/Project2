"""数据检查：训练前确认数据读取、划分和标签都正确。

检查项：各子集数量、标签范围、类别分布、MEAN/STD、空白图、跨子集重复图像；并画样例网格。
用法：python check_data.py        输出到 reports/data_check/
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import CLASSES, MEAN, SPLITS, STD, load_all

EXPECTED_COUNTS = {"Training": 28709, "PublicTest": 3589, "PrivateTest": 3589}
OUT_DIR = Path("reports/data_check")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    images, labels, usages = data["images"], data["labels"], data["usages"]
    lines = ["# 数据检查报告", ""]

    def check(ok: bool, text: str) -> None:
        lines.append(f"- {'✅' if ok else '❌'} {text}")
        print(("PASS " if ok else "FAIL ") + text)

    # 1. 数量与标签
    for split, expected in EXPECTED_COUNTS.items():
        n = int((usages == split).sum())
        check(n == expected, f"{split}: {n} 张（应为 {expected}）")
    check(set(np.unique(usages)) == set(SPLITS), f"Usage 取值: {sorted(set(usages.tolist()))}")
    check(labels.min() == 0 and labels.max() == 6, f"标签范围 {labels.min()}~{labels.max()}")
    check(images.shape[1:] == (48, 48), f"图像尺寸 {images.shape[1:]}")

    # 2. 归一化统计量（只用 Training）
    train_px = images[usages == "Training"] / 255.0
    mean, std = train_px.mean(), train_px.std()
    check(abs(mean - MEAN) < 1e-3 and abs(std - STD) < 1e-3,
          f"Training MEAN={mean:.4f} STD={std:.4f}（dataset.py 中为 {MEAN}/{STD}）")

    # 3. 空白图与跨子集重复（FER2013 已知存在少量，这里只统计、不修改官方划分）
    blank = int((images.reshape(len(images), -1).std(axis=1) < 1).sum())
    lines.append(f"- ℹ️ 几乎纯色的空白图：{blank} 张")
    hashes = np.array([hashlib.md5(img.tobytes()).hexdigest() for img in images])
    train_hashes = set(hashes[usages == "Training"])
    for split in ("PublicTest", "PrivateTest"):
        dup = sum(h in train_hashes for h in hashes[usages == split])
        lines.append(f"- ℹ️ {split} 中与 Training 像素完全相同的图：{dup} 张")

    # 4. 类别分布表 + 图
    lines += ["", "## 类别分布", "", "| 类别 | " + " | ".join(SPLITS) + " |",
              "|---|" + "---|" * len(SPLITS)]
    counts = {s: np.bincount(labels[usages == s], minlength=7) for s in SPLITS}
    for i, name in enumerate(CLASSES):
        lines.append(f"| {i} {name} | " + " | ".join(str(counts[s][i]) for s in SPLITS) + " |")

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.27
    for k, split in enumerate(SPLITS):
        ax.bar(np.arange(7) + (k - 1) * width, counts[split], width, label=split)
    ax.set_xticks(range(7), CLASSES)
    ax.set(ylabel="count", title="Class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 5. 样例网格：每类随机 8 张 Training 图，目视确认图像与标签对应
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(7, 8, figsize=(8, 7.5))
    for i, name in enumerate(CLASSES):
        idx = rng.choice(np.flatnonzero((labels == i) & (usages == "Training")), 8, replace=False)
        for j, k in enumerate(idx):
            axes[i, j].imshow(images[k], cmap="gray", vmin=0, vmax=255)
            axes[i, j].axis("off")
        axes[i, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "samples.png", dpi=120)
    plt.close(fig)

    lines += ["", "![分布](class_distribution.png)", "", "![样例](samples.png)"]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
