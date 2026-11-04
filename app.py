"""Gradio 演示：摄像头实时画框识别，或上传/拍照识别人脸表情。

实时模式：服务器只返回人脸框坐标和概率（几百字节），框由浏览器直接画在摄像头画面上，
不回传整张图片，延迟低；摄像头组件只作输入，录制/停止按钮状态稳定。

流程：RGB 图 → YuNet 检测人脸 → 按双眼转正并裁剪 → 灰度 48x48 → 与评估完全相同的预处理和推理 → 7 类概率。
前端只做演示，不参与评估；报告中的准确率全部来自 evaluate.py。

用法：
    python app.py                                  # 读取 runs/final_runs.txt 中的最终模型
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
LIVE_TITLE = "最大人脸的表情概率"


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


def prob_html(probs: np.ndarray | None, title: str = "识别结果") -> str:
    """7 类概率按类别编号 0–6 固定顺序显示（不按大小排序），最高的一类用强调色。"""
    probs = np.zeros(len(CLASSES)) if probs is None else probs
    top = int(probs.argmax()) if probs.any() else -1
    rows = "".join(
        f'<div class="prob-row{" top" if i == top else ""}">'
        f'<span class="name"><i>{i}</i><b>{zh}</b><small>{en}</small></span>'
        f'<span class="bar"><span style="width:{p * 100:.1f}%"></span></span>'
        f'<span class="pct">{p * 100:.1f}%</span></div>'
        for i, (zh, en, p) in enumerate(zip(CLASSES_ZH, CLASSES, probs, strict=True)))
    return f'<div class="probs"><div class="section-title">{title}</div>{rows}</div>'


def predict(models, rgb: np.ndarray | None):
    """上传/拍照模式：返回 (7 类概率 HTML, 模型实际看到的 48x48 人脸)。"""
    if rgb is None:
        return prob_html(None), None
    face, found = crop_face(rgb)
    if not found:
        gr.Warning("未检测到人脸，已使用整张图片，结果可能不准。")
    preview = cv2.resize(face, (192, 192), interpolation=cv2.INTER_NEAREST)
    return prob_html(ensemble_probs(models, [face])[0]), preview


def detect_live(models, rgb: np.ndarray | None):
    """实时模式：检测画面中的人脸（最多 5 张）并成批识别。

    返回 (人脸列表给浏览器画框, 最大人脸的 7 类概率 HTML)。坐标归一化到 0–1，
    与浏览器显示的画面一一对应（Gradio 发送的帧与预览一样做了镜像）。
    """
    if rgb is None:
        return {"faces": []}, prob_html(None, LIVE_TITLE)
    faces = detect_faces(rgb, max_width=DETECT_WIDTH)[:5]
    if not faces:
        return {"faces": []}, prob_html(None, LIVE_TITLE)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    probs = ensemble_probs(models, [align_crop(gray, box, eyes) for box, eyes in faces])
    h, w = gray.shape
    out = []
    for (box, _), p in zip(faces, probs, strict=True):
        top = int(p.argmax())
        out.append({"box": [float(box[0] / w), float(box[1] / h), float(box[2] / w), float(box[3] / h)],
                    "text": f"{CLASSES[top]} {p[top] * 100:.0f}%"})
    return {"faces": out}, prob_html(probs[0], LIVE_TITLE)


# 浏览器端画框：在摄像头 <video> 上叠一层 canvas。超过 1 秒没有新结果（停止录制）就清空。
DRAW_JS = """
(data) => {
  const video = document.querySelector('#live-cam video');
  if (!video) return;
  const host = video.parentElement;
  let cv = host.querySelector('canvas.face-overlay');
  if (!cv) {
    cv = document.createElement('canvas');
    cv.className = 'face-overlay';
    host.style.position = 'relative';
    host.appendChild(cv);
  }
  const vr = video.getBoundingClientRect(), hr = host.getBoundingClientRect();
  Object.assign(cv.style, {left: (vr.left - hr.left) + 'px', top: (vr.top - hr.top) + 'px',
                           width: vr.width + 'px', height: vr.height + 'px'});
  cv.width = vr.width; cv.height = vr.height;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  clearTimeout(window.ferClear);
  window.ferClear = setTimeout(() => ctx.clearRect(0, 0, cv.width, cv.height), 1000);
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !data || !data.faces) return;
  const fit = getComputedStyle(video).objectFit;   // 视频在元素里的实际显示区域
  const s = fit === 'cover' ? Math.max(vr.width / vw, vr.height / vh) : Math.min(vr.width / vw, vr.height / vh);
  const ox = (vr.width - vw * s) / 2, oy = (vr.height - vh * s) / 2;
  ctx.lineWidth = 2; ctx.font = '500 14px "DM Sans", sans-serif'; ctx.textBaseline = 'top';
  for (const f of data.faces) {
    const [x, y, w, h] = f.box;
    const X = ox + x * vw * s, Y = oy + y * vh * s, W = w * vw * s, H = h * vh * s;
    ctx.strokeStyle = '#d97757'; ctx.strokeRect(X, Y, W, H);
    const tw = ctx.measureText(f.text).width + 8, ty = Math.max(0, Y - 20);
    ctx.fillStyle = '#d97757'; ctx.fillRect(X - 1, ty, tw, 20);
    ctx.fillStyle = '#fff'; ctx.fillText(f.text, X + 3, ty + 3);
  }
}
"""


# ---------------------------------------------------------------- 界面
def model_info(runs: list[Path]) -> str:
    """页头信息：测试集准确率只读取 evaluate.py 生成的报告，前端不做任何评估。"""
    name = "PrivateTest" if len(runs) == 1 else f"PrivateTest_ensemble{len(runs)}"
    report = Path("reports") / name / "metrics.json"
    acc = "这组模型没有 PrivateTest 结果"
    if report.exists():
        metrics = json.loads(report.read_text())
        # 只有报告里的模型和当前加载的模型完全相同时才显示，避免把别的模型的成绩挂上来
        if sorted(Path(m["run"]).name for m in metrics["members"]) == sorted(r.name for r in runs):
            acc = f"PrivateTest 准确率 {metrics['accuracy'] * 100:.2f}%"
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


# 配色参考 Claude：暖米白底、陶土橙强调、衬线标题；暗色模式单独配色而不是简单反色。
CLAY = gr.themes.Color(  # 陶土橙色阶（c500 为主色 #d97757）
    c50="#fbf1ec", c100="#f6e0d6", c200="#efc2ae", c300="#e7a387", c400="#e08b6c",
    c500="#d97757", c600="#c96442", c700="#a94f33", c800="#853f2a", c900="#5f2e1f", c950="#3d1d14",
    name="clay")

THEME = gr.themes.Base(
    primary_hue=CLAY,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_lg,
    font=(gr.themes.GoogleFont("DM Sans", weights=(400, 500, 600)), "Microsoft YaHei", "PingFang SC",
          "sans-serif"),
).set(
    body_background_fill="#faf9f5", body_background_fill_dark="#262624",
    body_text_color="#141413", body_text_color_dark="#f5f4ee",
    body_text_color_subdued="#73726c", body_text_color_subdued_dark="#a6a39a",
    background_fill_primary="#ffffff", background_fill_primary_dark="#30302e",
    background_fill_secondary="#f5f4ed", background_fill_secondary_dark="#2b2b29",
    block_background_fill="#ffffff", block_background_fill_dark="#30302e",
    block_border_color="#e8e6dc", block_border_color_dark="#3e3d39",
    border_color_primary="#e8e6dc", border_color_primary_dark="#3e3d39",
    block_shadow="none", block_shadow_dark="none",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_label_text_color="#73726c", block_label_text_color_dark="#a6a39a",
    block_label_border_width="0px",
    button_primary_background_fill="#c96442", button_primary_background_fill_dark="#c96442",
    button_primary_background_fill_hover="#b5573a", button_primary_background_fill_hover_dark="#d97757",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_primary_border_color="transparent", button_primary_border_color_dark="transparent",
    button_secondary_background_fill="#f0eee6", button_secondary_background_fill_dark="#3a3936",
    button_secondary_background_fill_hover="#e8e6dc", button_secondary_background_fill_hover_dark="#45443f",
    button_secondary_text_color="#141413", button_secondary_text_color_dark="#f5f4ee",
    input_background_fill="#ffffff", input_background_fill_dark="#30302e",
    color_accent="#d97757", color_accent_soft="#f6e0d6", color_accent_soft_dark="#4a3229",
    link_text_color="#c96442", link_text_color_dark="#e08b6c",
)

CSS = """
@import url("https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap");
:root {--fer-track: #ece9df; --fer-fill: #c2bdb0; --fer-muted: #73726c; --fer-accent: #d97757;}
.dark {--fer-track: #3a3936; --fer-fill: #6b6962; --fer-muted: #a6a39a; --fer-accent: #d97757;}
#header {padding: 28px 4px 8px;}
#header h1 {font-family: "Source Serif 4", Georgia, "Songti SC", "Microsoft YaHei", serif; font-weight: 500;
            font-size: 2.1rem; letter-spacing: -0.01em; margin: 0;}
#header h1 .mark {color: var(--fer-accent); margin-right: 0.3em;}
#header p {color: var(--fer-muted); margin: 6px 0 0; font-size: 0.95rem;}
.section-title {font-family: "Source Serif 4", Georgia, serif; font-size: 1.05rem; font-weight: 500;
                margin: 0 0 4px;}
.gradio-container {width: 100% !important; max-width: 1200px !important; margin: 0 auto !important;}
canvas.face-overlay {position: absolute; pointer-events: none; z-index: 5;}
.probs {display: flex; flex-direction: column; gap: 14px; padding: 12px 6px;}
.prob-row {display: grid; grid-template-columns: 9em 1fr 3.6em; align-items: center; gap: 14px;
           color: var(--fer-muted);}
.prob-row .name b {font-weight: 500; color: var(--body-text-color); margin-right: 0.35em;}
.prob-row .name small {font-size: 0.8em;}
.prob-row .name i {font-style: normal; font-size: 0.8em; margin-right: 0.6em; font-variant-numeric: tabular-nums;}
.prob-row .bar {height: 6px; background: var(--fer-track); border-radius: 999px; overflow: hidden;}
.prob-row .bar span {display: block; height: 100%; border-radius: 999px; background: var(--fer-fill);
                     transition: width 0.25s ease;}
.prob-row.top .name b, .prob-row.top .pct {color: var(--fer-accent); font-weight: 600;}
.prob-row.top .bar span {background: var(--fer-accent);}
.prob-row .pct {text-align: right; font-variant-numeric: tabular-nums;}
"""

def build_app(models, runs: list[Path]) -> gr.Blocks:
    with gr.Blocks(title="人脸表情识别") as demo:
        gr.HTML(f'<div id="header"><h1><span class="mark">✻</span>人脸表情识别</h1>'
                f'<p>{model_info(runs)}</p></div>')
        with gr.Tab("实时摄像头"), gr.Row(equal_height=False):
            with gr.Column(scale=3):
                # 摄像头只作输入（Gradio 官方推荐的分离写法）；把它同时作输出会让录制/停止按钮来回闪
                cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", elem_id="live-cam",
                               label="摄像头（点击录制开始实时识别）",
                               webcam_options=gr.WebcamOptions(constraints={"video": {
                                   "width": LIVE_SIZE[0], "height": LIVE_SIZE[1]}}))
            with gr.Column(scale=2):
                live_probs = gr.HTML(prob_html(None, LIVE_TITLE))
                live_faces = gr.JSON(visible="hidden")
        with gr.Tab("上传 / 拍照"), gr.Row(equal_height=False):
            with gr.Column(scale=1):
                image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="输入图像")
                run_btn = gr.Button("识别表情", variant="primary")
                gr.Examples(make_examples(), inputs=image, label="示例（PublicTest，每类一张）")
            with gr.Column(scale=1):
                label = gr.HTML(prob_html(None))
                face = gr.Image(label="模型看到的人脸（48×48）", height=200, interactive=False)

        cam.stream(lambda rgb: detect_live(models, rgb), inputs=cam, outputs=[live_faces, live_probs],
                   stream_every=0.1, concurrency_limit=1, show_progress="hidden")
        live_faces.change(None, inputs=live_faces, js=DRAW_JS, show_progress="hidden")
        run_btn.click(lambda rgb: predict(models, rgb), inputs=image, outputs=[label, face])
        image.clear(lambda: (prob_html(None), None), outputs=[label, face])
    return demo


DEVICE = get_device()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人脸表情识别演示")
    parser.add_argument("--runs", type=Path, nargs="+", default=None,
                        help="模型目录；缺省时读取 runs/final_runs.txt")
    parser.add_argument("--single", action="store_true",
                   help="使用 runs/best_single.txt 里的最佳单模型（缺省用 runs/final_runs.txt 的多模型平均）")
    parser.add_argument("--share", action="store_true", help="生成临时公网链接")
    parser.add_argument("--inbrowser", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args()
    args.runs = args.runs or final_runs(single=args.single)
    torch.set_grad_enabled(False)
    print(f"推理设备: {DEVICE}")
    try:
        build_app(load_runs(args.runs, DEVICE), args.runs).launch(
            theme=THEME, css=CSS, share=args.share, inbrowser=args.inbrowser)
    except KeyboardInterrupt:  # 关闭服务器时又按了一次 Ctrl+C，直接安静退出
        print("已退出。")
