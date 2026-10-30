"""Gradio 演示：摄像头实时画框识别，或上传/拍照识别人脸表情。

流程：RGB 图 → YuNet 检测人脸 → 按双眼转正并裁剪 → 灰度 48x48 → 与评估完全相同的预处理和推理 → 7 类概率。
前端只做演示，不参与评估；报告中的准确率全部来自 evaluate.py。

用法：
    python app.py                                  # 读取 final_runs.txt 中的最终模型
    python app.py --runs runs/A1_s0 runs/A2_s0     # 指定模型（多个即集成）
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from dataset import CLASSES, CLASSES_ZH, IMG_SIZE, load_split, preprocess_face
from utils import final_runs, get_device, load_runs, model_probs

EXAMPLES_DIR = Path("examples")
YUNET_PATH = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
DETECT_LOCK = threading.Lock()  # YuNet 检测器不是线程安全的
DETECT_WIDTH = 320  # 实时模式下在缩小的画面上检测人脸，再把框映射回原图，速度快很多
LIVE_SIZE = (640, 480)  # 向浏览器请求的摄像头分辨率，传输量小、延迟低
BOX_COLOR = (234, 88, 12)  # 画框颜色（RGB）
LABELS = [f"{zh} {en}" for zh, en in zip(CLASSES_ZH, CLASSES, strict=True)]


# ---------------------------------------------------------------- 人脸检测与裁剪
def make_yunet():
    """OpenCV 自带的 YuNet 深度学习人脸检测器（MIT 许可），比 Haar 更能应对侧脸、眼镜、近距离。"""
    if not YUNET_PATH.exists():
        return None
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold=0.6)


YUNET = make_yunet()


def detect_faces(rgb: np.ndarray, max_width: int | None = None) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """返回 [(框 x,y,w,h, 双眼坐标 或 None)]，按面积从大到小，坐标对应原图。

    max_width：先把画面缩到这个宽度再检测（实时模式用）；优先 YuNet，模型文件缺失时退回 Haar。
    """
    scale = 1.0
    small = rgb
    if max_width and rgb.shape[1] > max_width:
        scale = rgb.shape[1] / max_width
        small = cv2.resize(rgb, (max_width, round(rgb.shape[0] / scale)), interpolation=cv2.INTER_AREA)
    if YUNET is not None:
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        with DETECT_LOCK:
            YUNET.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, dets = YUNET.detect(bgr)
        faces = [] if dets is None else [(d[:4] * scale, d[4:8].reshape(2, 2) * scale) for d in dets]
    else:
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        boxes = HAAR.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        faces = [(np.array(b, float) * scale, None) for b in boxes]
    return sorted(faces, key=lambda f: -f[0][2] * f[0][3])


def align_crop(gray: np.ndarray, box: np.ndarray, eyes: np.ndarray | None) -> np.ndarray:
    """按双眼把脸转正，再以人脸中心取正方形（边长为框的 1.2 倍，接近 FER2013 的裁剪），缩放到 48x48。"""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    angle = 0.0
    if eyes is not None:  # YuNet 关键点：右眼、左眼（图像坐标）
        (rx, ry), (lx, ly) = eyes
        angle = float(np.degrees(np.arctan2(ly - ry, lx - rx)))
    scale = IMG_SIZE / (1.2 * max(w, h))
    mat = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, scale)
    mat[:, 2] += (IMG_SIZE / 2 - cx, IMG_SIZE / 2 - cy)
    return cv2.warpAffine(gray, mat, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


def crop_face(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """返回 (48x48 灰度人脸, 是否检测到人脸)。输入本身是 48x48 时直接使用（便于与测试集对照）。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    if gray.shape == (IMG_SIZE, IMG_SIZE):
        return gray, True
    rgb3 = rgb if rgb.ndim == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    faces = detect_faces(rgb3)
    if faces:
        return align_crop(gray, *faces[0]), True
    return cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), False


# ---------------------------------------------------------------- 推理
def ensemble_probs(models, faces: list[np.ndarray]) -> np.ndarray:
    """多张人脸一次成批推理，返回 (人脸数, 7) 的集成概率。"""
    x = torch.cat([preprocess_face(f) for f in faces]).to(DEVICE)
    return (sum(model_probs(m, x, tta) for m, tta, _ in models) / len(models)).cpu().numpy()


def to_label(probs: np.ndarray) -> dict[str, float]:
    return {name: float(p) for name, p in zip(LABELS, probs, strict=True)}


def predict(models, rgb: np.ndarray | None):
    """上传/拍照模式：返回 (7 类概率, 模型实际看到的 48x48 人脸)。"""
    if rgb is None:
        return None, None
    face, found = crop_face(rgb)
    if not found:
        gr.Warning("未检测到人脸，已使用整张图片，结果可能不准。")
    preview = cv2.resize(face, (192, 192), interpolation=cv2.INTER_NEAREST)
    return to_label(ensemble_probs(models, [face])[0]), preview


def annotate(models, rgb: np.ndarray | None):
    """实时模式：检测画面中的人脸（最多 5 张），直接在摄像头画面上画框并标出表情。

    返回 (标注后的画面, 最大人脸的 7 类概率)。
    """
    if rgb is None:
        return None, None
    faces = detect_faces(rgb, max_width=DETECT_WIDTH)[:5]
    if not faces:
        return rgb, None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    probs = ensemble_probs(models, [align_crop(gray, box, eyes) for box, eyes in faces])
    out = rgb.copy()
    thick = max(2, rgb.shape[1] // 320)
    font = 0.45 * thick
    for (box, _), p in zip(faces, probs, strict=True):
        x, y, w, h = box.astype(int)
        top = int(p.argmax())
        text = f"{CLASSES[top]} {p[top] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        ty = max(y, th + 8)
        cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, thick)
        cv2.rectangle(out, (x, ty - th - 8), (x + tw + 8, ty), BOX_COLOR, -1)
        cv2.putText(out, text, (x + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return out, to_label(probs[0])


# ---------------------------------------------------------------- 界面
def model_info(runs: list[Path]) -> str:
    """页头信息：测试集准确率只读取 evaluate.py 生成的报告，前端不做任何评估。"""
    name = "PrivateTest" if len(runs) == 1 else f"PrivateTest_ensemble{len(runs)}"
    report = Path("reports") / name / "metrics.json"
    acc = f"PrivateTest 准确率 {json.loads(report.read_text())['accuracy'] * 100:.2f}%" \
        if report.exists() else "PrivateTest 尚未评估"
    kind = "ResNet18" if len(runs) == 1 else f"{len(runs)} × ResNet18 集成"
    return f"{kind} · FER2013 从头训练 · {acc}"


def make_examples() -> list[str]:
    """从 PublicTest（验证集）每类取一张图作为示例；不使用 PrivateTest。"""
    EXAMPLES_DIR.mkdir(exist_ok=True)
    images, labels = load_split("PublicTest")
    paths = []
    for i, name in enumerate(CLASSES):
        path = EXAMPLES_DIR / f"{i}_{name}.png"
        if not path.exists():
            cv2.imwrite(str(path), images[np.flatnonzero(labels == i)[0]])
        paths.append(str(path))
    return paths


CSS = ".live {max-width: 720px; margin: 0 auto;}"  # 实时页居中

THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font=(gr.themes.GoogleFont("Inter"), "Microsoft YaHei", "PingFang SC", "sans-serif"),
)


def build_app(models, runs: list[Path]) -> gr.Blocks:
    with gr.Blocks(title="人脸表情识别") as demo:
        gr.Markdown(f"# 人脸表情识别\n{model_info(runs)}")
        with gr.Tab("实时摄像头"), gr.Column(elem_classes="live"):
            # Gradio 官方“Unified Image”写法：摄像头组件同时作为输入和输出，框直接画在摄像头画面上
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="摄像头（点击录制开始实时识别）",
                           webcam_options=gr.WebcamOptions(constraints={"video": {
                               "width": LIVE_SIZE[0], "height": LIVE_SIZE[1]}}))
            live_label = gr.Label(num_top_classes=3, label="最大人脸的表情")
        with gr.Tab("上传 / 拍照"), gr.Row(equal_height=False):
            with gr.Column(scale=1):
                image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="输入图像")
                run_btn = gr.Button("识别表情", variant="primary")
                gr.Examples(make_examples(), inputs=image, label="示例（PublicTest，每类一张）")
            with gr.Column(scale=1):
                label = gr.Label(num_top_classes=7, label="识别结果")
                face = gr.Image(label="模型看到的人脸（48×48）", height=200, interactive=False)

        cam.stream(lambda rgb: annotate(models, rgb), inputs=cam, outputs=[cam, live_label],
                   stream_every=0.1, concurrency_limit=1, show_progress="hidden")
        run_btn.click(lambda rgb: predict(models, rgb), inputs=image, outputs=[label, face])
        image.clear(lambda: (None, None), outputs=[label, face])
    return demo


DEVICE = get_device()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人脸表情识别演示")
    parser.add_argument("--runs", type=Path, nargs="+", default=None,
                        help="模型目录；缺省时读取 final_runs.txt")
    parser.add_argument("--share", action="store_true", help="生成临时公网链接")
    parser.add_argument("--inbrowser", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args()
    args.runs = args.runs or final_runs()
    torch.set_grad_enabled(False)
    print(f"推理设备: {DEVICE}")
    build_app(load_runs(args.runs, DEVICE), args.runs).launch(
        theme=THEME, css=CSS, share=args.share, inbrowser=args.inbrowser)
