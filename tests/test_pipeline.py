"""训练前的快速检查（CPU 上约 1 分钟）：python -m pytest -q tests"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402
import train  # noqa: E402
from model import build_model  # noqa: E402
from utils import make_loader  # noqa: E402


def test_split_sizes_and_labels():
    for split, n in (("Training", 28709), ("PublicTest", 3589), ("PrivateTest", 3589)):
        images, labels = dataset.load_split(split)
        assert images.shape == (n, 48, 48) and images.dtype == np.uint8
        assert labels.min() >= 0 and labels.max() <= 6


def test_mean_std_from_training_only():
    images, _ = dataset.load_split("Training")
    px = images / 255.0
    assert abs(px.mean() - dataset.MEAN) < 1e-3
    assert abs(px.std() - dataset.STD) < 1e-3


def test_model_output_shape():
    model = build_model().eval()
    assert model(torch.zeros(4, 1, 48, 48)).shape == (4, 7)


def test_eval_preprocessing_is_shared_with_app():
    """前端 preprocess_face 与验证/测试 DataLoader 得到的张量必须逐元素相同。"""
    images, labels = dataset.load_split("PublicTest")
    x_loader, _ = next(iter(make_loader(images[:8], labels[:8], train=False, batch_size=8)))
    x_app = torch.cat([dataset.preprocess_face(img) for img in images[:8]])
    assert torch.equal(x_loader, x_app)
    assert abs(x_loader.mean().item()) < 0.5  # 已归一化到 0 附近


def test_train_never_reads_private_test_and_resumes(tmp_path, monkeypatch):
    """训练只读取 Training/PublicTest；中断后 --resume 能接着上次的轮数继续。"""
    seen = []
    real_load_split = dataset.load_split

    def spy(split):
        seen.append(split)
        return real_load_split(split)

    monkeypatch.setattr(train, "load_split", spy)
    base = ["train.py", "--exp", "A3", "--subset", "32", "--batch-size", "16", "--workers", "0",
            "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", base + ["--epochs", "1"])
    train.main()
    monkeypatch.setattr(sys, "argv", base + ["--epochs", "2", "--resume"])
    train.main()

    assert "PrivateTest" not in seen and set(seen) == {"Training", "PublicTest"}
    run_dir = tmp_path / "A3_s0"
    with open(run_dir / "log.csv", newline="") as f:
        assert [int(r["epoch"]) for r in csv.DictReader(f)] == [1, 2]
    for name in ("best.pt", "last.pt", "config.json", "summary.json", "curves.png"):
        assert (run_dir / name).exists()


def test_private_test_only_in_evaluate():
    """源码层面：除 dataset.py 的定义和 evaluate.py 外，任何脚本都不出现 PrivateTest 字样。"""
    root = Path(__file__).resolve().parents[1]
    allowed = {"dataset.py", "evaluate.py", "check_data.py", "app.py"}
    for path in root.glob("*.py"):
        if path.name not in allowed:
            assert "PrivateTest" not in path.read_text(encoding="utf-8"), path.name


def test_app_matches_evaluate():
    """前端单张推理与 evaluate.py 的批量推理结果一致（需要已训练的 runs/*/best.pt，否则跳过）。"""
    import pytest

    runs = sorted(Path(__file__).resolve().parents[1].glob("runs/*/summary.json"))
    if not runs:
        pytest.skip("还没有训练好的模型")
    import app
    from utils import load_runs, predict_probs

    model, tta, _ = load_runs([runs[0].parent], app.DEVICE)[0]
    images, labels = dataset.load_split("PublicTest")
    images, labels = images[:32], labels[:32]
    p_eval, _ = predict_probs(model, make_loader(images, labels, train=False, batch_size=32),
                              app.DEVICE, tta=tta)
    faces = [app.crop_face(img)[0] for img in images]
    x_app = torch.cat([dataset.preprocess_face(f) for f in faces]).to(app.DEVICE)
    p_app = app.model_probs(model, x_app, tta).cpu()
    assert torch.allclose(p_app, p_eval, atol=1e-5)
