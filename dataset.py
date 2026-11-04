"""数据层：读取 FER2013 CSV、按 Usage 划分、预处理与数据增强。

数据协议（全项目只在这里定义一次）：
    Training   -> 只用于更新模型参数
    PublicTest -> 验证集：早停、选 checkpoint、比较实验、决定是否 TTA
    PrivateTest-> 测试集：只在 evaluate.py 中做最终评估
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_NAMES = ("icml_face_data.csv", "fer2013.csv")  # 两种常见文件名，列顺序不同也能读
CACHE_PATH = DATA_DIR / "fer2013.npz"  # 首次解析 CSV 后的缓存，之后加载只需约 1 秒

SPLITS = ("Training", "PublicTest", "PrivateTest")
CLASSES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
CLASSES_ZH = ("愤怒", "厌恶", "恐惧", "快乐", "悲伤", "惊讶", "中性")
IMG_SIZE = 48

# 只用 Training 集像素（除以 255 后）统计得到；训练、评估、前端共用这一组数值。
# tests/test_pipeline.py 会重新计算并核对。
MEAN = 0.5077
STD = 0.2550


def _parse_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """逐行解析 CSV，检查每行格式，返回全部图像、标签和 Usage。"""
    images, labels, usages = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        if sorted(header) != ["Usage", "emotion", "pixels"]:
            raise ValueError(f"CSV 列名应为 emotion、Usage、pixels（顺序不限），实际为 {header}")
        col = {name: header.index(name) for name in header}  # 按列名取值，不依赖列顺序
        rows = tqdm(reader, desc="解析 CSV", unit=" 行", total=35887)
        for line_no, row in enumerate(rows, start=2):
            emotion, usage, pixels = row[col["emotion"]], row[col["Usage"]], row[col["pixels"]]
            pix = np.array(pixels.split(), dtype=np.uint8)
            if pix.size != IMG_SIZE * IMG_SIZE:
                raise ValueError(f"第 {line_no} 行像素数为 {pix.size}，应为 2304")
            images.append(pix.reshape(IMG_SIZE, IMG_SIZE))
            labels.append(int(emotion))
            usages.append(usage.strip())
    return {
        "images": np.stack(images),                  # (N, 48, 48) uint8
        "labels": np.array(labels, dtype=np.int64),  # (N,)
        "usages": np.array(usages),                  # (N,) str
    }


def find_csv() -> Path:
    """在 data/ 里找数据 CSV：icml_face_data.csv 或 fer2013.csv。"""
    for name in CSV_NAMES:
        if (DATA_DIR / name).exists():
            return DATA_DIR / name
    raise FileNotFoundError(f"没有找到数据文件：请把 {' 或 '.join(CSV_NAMES)} 放进 {DATA_DIR}")


def load_all(csv_path: Path | None = None, cache_path: Path = CACHE_PATH) -> dict[str, np.ndarray]:
    """读取全部数据（优先读缓存）。load_split 和 check_data.py 通过它取数据。"""
    if cache_path.exists():
        with np.load(cache_path) as d:
            return {k: d[k] for k in d.files}
    csv_path = csv_path or find_csv()
    print(f"首次运行：解析 {csv_path.name} 并缓存到 {cache_path.name}（只需一次，约 1 分钟）")
    data = _parse_csv(csv_path)
    print("正在写入缓存 ...")
    np.savez_compressed(cache_path, **data)
    print("缓存完成，之后加载约 1 秒")
    return data


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """返回某个子集的 (images, labels)。所有脚本都通过这个函数取数据。"""
    if split not in SPLITS:
        raise ValueError(f"未知子集 {split!r}，可选 {SPLITS}")
    data = load_all()
    mask = data["usages"] == split
    return data["images"][mask], data["labels"][mask]


def build_transform(train: bool, random_erasing: bool = False) -> v2.Compose:
    """输入 (1, 48, 48) uint8 张量，输出归一化后的 float32 张量。

    train=False 的变换是"标准预处理"，验证、测试、前端都用它，保证三处一致。
    """
    to_input = [v2.ToDtype(torch.float32, scale=True), v2.Normalize([MEAN], [STD])]
    if not train:
        return v2.Compose(to_input)
    augment = [
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ]
    erase = [v2.RandomErasing(p=0.25)] if random_erasing else []
    return v2.Compose(augment + to_input + erase)


class FERDataset(Dataset):
    """把 (N, 48, 48) uint8 图像包装成 PyTorch Dataset。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: v2.Compose):
        self.images = torch.from_numpy(images).unsqueeze(1)  # (N, 1, 48, 48)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.images[i]), self.labels[i]


def preprocess_face(gray48: np.ndarray) -> torch.Tensor:
    """前端用：把一张 48x48 uint8 灰度图变成模型输入 (1, 1, 48, 48)，与评估流程完全一致。"""
    if gray48.shape != (IMG_SIZE, IMG_SIZE) or gray48.dtype != np.uint8:
        raise ValueError("需要 48x48 的 uint8 灰度图")
    return build_transform(train=False)(torch.from_numpy(gray48)[None]).unsqueeze(0)
