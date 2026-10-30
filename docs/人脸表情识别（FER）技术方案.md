# 人脸表情识别（FER）技术方案

FER2013 · PyTorch · 单人实现 · 2026-10-27



1. 项目通知（PDF）：包含项目要求、数据集说明、评分标准，请务必仔细阅读；

2. 人脸表情数据集（CSV）（因为文件大小问题提交在ajoubb课程文件中）：包含 emotion、pixels、Usage 三个字段，请按 Usage 字段划分数据：

   · Training（28,709 张）：用于模型训练；

   · PublicTest（3,589 张）：作为验证集，用于调参和模型选择；

   · PrivateTest（3,589 张）：作为测试集，仅用于最终评估，严禁用于训练或调参。



## 1. 任务与约束

- 任务：48×48 灰度人脸图像的 7 分类（0 愤怒、1 厌恶、2 恐惧、3 快乐、4 悲伤、5 惊喜、6 中立），只判断表情，不识别身份。
- 数据：CSV，字段 emotion、pixels、Usage；Training 28,709 / PublicTest 3,589 / PrivateTest 3,589。
- 数据协议（必须落实到代码结构里）：Training 只用于更新参数；PublicTest 用于所有选择（结构、超参数、早停、checkpoint、集成权重、是否使用 TTA）；PrivateTest 只在最终评估脚本中读取一次。
- 交付物：可运行代码、逐轮 loss/acc 日志、最佳 checkpoint、PrivateTest 准确率、混淆矩阵。

## 2. 总体方案

流程：CSV → 数据检查 → Dataset 与增强 → 改造版 ResNet18 → SGD（Nesterov）+ ReduceLROnPlateau 训练，按 PublicTest 选模型 → 加载最佳 checkpoint → TTA → PrivateTest 单次评估 → 混淆矩阵与分析。

预期指标（经验估计，需自行实验验证）：

| 阶段                      | 预期 PrivateTest 准确率 |
| ----------------------- | ------------------ |
| 规范基线（ResNet18 + 增强）     | 约 68%\~72%         |
| + TTA、label smoothing 等 | 约 72%\~74%         |
| + 2\~3 个模型集成            | 再提升约 1\~2 点        |

文献参照（开源项目与论文自报，同一数据集、不使用额外训练数据）：VGG 精细调参 73.28%，ResNet18 加训练技巧 73.70%，ResMaskingNet 单模型 74.14%，6 模型集成 76.82%。人类水平约 65±5%，FER2013 标签噪声大，单网络 73%\~74% 已接近实用上限。

## 3. 环境

```
python -m venv fer_env
fer_env\\Scripts\\activate
pip install torch torchvision   # 有 NVIDIA 显卡时按 pytorch.org 给出的命令安装 CUDA 版
pip install numpy pandas matplotlib scikit-learn tqdm
pip freeze > requirements.txt
```

不需要 TensorFlow、OpenCV、Jupyter。外层封装库（Lightning、Hydra 等）不使用，训练循环自己写；超参数用标准库 argparse 传入；日志写 CSV，用 matplotlib 画曲线。

## 4. 项目结构

```
fer/
  data/fer.csv
  dataset.py     # 读取 CSV、Dataset、增强
  model.py       # 网络定义
  train.py       # 训练与按 PublicTest 选模型
  evaluate.py    # 加载 checkpoint，PrivateTest 单次评估
  utils.py       # 随机种子、日志、画图
  runs/          # 日志 CSV、曲线、best.pt
  requirements.txt
```

## 5. 数据层

### 5.1 读取后必须检查

- Usage 取值与数量是否为 28709 / 3589 / 3589
- emotion 取值是否在 0\~6，无缺失
- 每行 pixels 是否正好 2304 个数值
- 各类别样本数（Disgust 仅数百张，Happy 近九千张）
- 随机画一张图片网格，目视确认图像与标签正常

### 5.2 Dataset

```python
import numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FER(Dataset):
    def __init__(self, csv_path, usage, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['Usage'] == usage]
        self.x = np.stack([np.array(p.split(), dtype=np.uint8).reshape(48, 48)
                           for p in df['pixels']])
        self.y = df['emotion'].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = Image.fromarray(self.x[i])
        if self.transform:
            img = self.transform(img)
        return img, int(self.y[i])
```

### 5.3 预处理与增强

MEAN、STD 只用 Training 集像素（除以 255 后）统计，三个子集共用同一组数值。验证集和测试集只做 ToTensor 与 Normalize，不做随机增强。

```python
from torchvision import transforms as T

train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.ToTensor(),
    T.Normalize([MEAN], [STD]),
    T.RandomErasing(p=0.25),      # 可选，作为消融项
])
eval_tf = T.Compose([T.ToTensor(), T.Normalize([MEAN], [STD])])
```

增强强度不宜过大：旋转超过约 15° 或平移过大会让表情特征失真，训练准确率反而上不去。

### 5.4 类别不均衡

评价指标是整体准确率，默认不做特殊处理。类别权重或加权采样只作为消融实验，观察对整体准确率与 Disgust 召回的影响。

## 6. 模型

选择：ResNet18，从头训练（不加载预训练权重）。对 48×48 灰度小图做三处改造：

```python
import torch.nn as nn
from torchvision.models import resnet18

def build_model(num_classes=7, p=0.4):
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)   # 单通道输入，3×3 卷积，stride=1
    m.maxpool = nn.Identity()                          # 去掉首个 maxpool，避免过早下采样
    m.fc = nn.Sequential(nn.Dropout(p), nn.Linear(512, num_classes))
    return m
```

特征图尺寸依次为 48 → 24 → 12 → 6，最后全局平均池化。若显存或速度紧张，可保留 maxpool（计算量约降为四分之一，精度通常略降）。备选结构：VGG 风格网络（4 个卷积块 + 两层全连接 + Dropout），参考 Khaireddin 等的做法。

## 7. 训练

| 项          | 取值                                                                 |
| ---------- | ------------------------------------------------------------------ |
| 损失         | CrossEntropy，label\_smoothing=0.1（作为消融项）                           |
| 优化器        | SGD，lr=0.01，momentum=0.9，nesterov=True，weight\_decay=5e-4          |
| 调度器        | ReduceLROnPlateau，mode=max，监控 PublicTest 准确率，factor=0.5，patience=5 |
| batch size | 128                                                                |
| epoch      | 最多 100\~200，PublicTest 准确率连续 20 轮不升则早停                             |
| 精度         | 混合精度（GPU 上）                                                        |
| 随机种子       | 固定 random、numpy、torch、cuda，并写入日志                                   |

核心循环：

```python
use_amp = dev.type == 'cuda'
model = build_model().to(dev)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      nesterov=True, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

def run_epoch(loader, train):
    model.train(train)
    tot_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.set_grad_enabled(train), torch.autocast(device_type=dev.type, enabled=use_amp):
            out = model(x)
            loss = crit(out, y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        tot_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return tot_loss / n, correct / n

best, bad = 0.0, 0
for ep in range(epochs):
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)        # 只用 PublicTest
    sched.step(va[1])
    log_row(ep, tr, va, opt.param_groups[0]['lr'])   # 写入 CSV
    if va[1] > best:
        best, bad = va[1], 0
        torch.save({'model': model.state_dict(), 'val_acc': best, 'args': vars(args)}, 'runs/best.pt')
    else:
        bad += 1
        if bad >= 20:
            break
```

日志列：epoch、train\_loss、train\_acc、val\_loss、val\_acc、lr。训练结束后读取 CSV 画 loss 曲线与 acc 曲线。注意：开启增强和 label smoothing 时，训练 loss 与训练准确率都是在带扰动的条件下算出的，数值会低于验证条件，这是正常现象。

参考开源实现时注意：WuJie1010 的 mainpro\_FER.py 按 PrivateTest 准确率保存最佳 checkpoint，直接使用会让测试集参与模型选择，必须改为按 PublicTest。

## 8. 评估

评估脚本独立于训练脚本，加载 best.pt，只读一次 PrivateTest。

```python
ck = torch.load('runs/best.pt', map_location=dev)
model.load_state_dict(ck['model'])
model.eval()
probs, ys = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(dev)
        p = model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)   # 翻转 TTA
        probs.append(p.cpu())
        ys.append(y)
pred = torch.cat(probs).argmax(1)
y = torch.cat(ys)
print('PrivateTest acc:', (pred == y).float().mean().item())
# sklearn.metrics.confusion_matrix / classification_report 输出混淆矩阵与各类召回
```

评估流程里不得出现任何反向传播、阈值或权重的调整。是否启用 TTA、集成权重取多少，都已在 PublicTest 上确定。

## 9. 提升手段（按性价比排序）

1. 测试时增强：水平翻转取概率平均；进阶用多裁剪（如 44×44 十个裁剪）。零训练成本，通常提升约 0.5\~1 点。
2. Label smoothing：已在训练配方中，作为消融项验证。
3. 学习率策略：ReduceLROnPlateau 为稳健基线；可对比 cosine annealing。
4. 更强正则：RandomErasing、Mixup 或 CutMix。小数据集上有时有益，也可能拖慢收敛，必须靠消融判断。
5. 模型集成：训练 2\~3 个不同种子或不同结构（如 ResNet18 + VGG 风格）的模型，对 softmax 概率取平均。每个模型各自按 PublicTest 选 checkpoint，集成方式也只用 PublicTest 评估。通常再提升约 1\~2 点，训练时间成倍增加。
6. 类别权重：只用于观察对少数类召回的影响，整体准确率可能下降。
7. 结构升级（如带注意力的 ResMaskingNet）：文献中单模型 74.14%，但实现复杂、训练慢，收益相对有限，不作为首选。

## 10. 实验设计

| 编号  | 配置                                |
| --- | --------------------------------- |
| A0  | 基线：ResNet18，无增强，无 label smoothing |
| A1  | A0 + 翻转与仿射增强                      |
| A2  | A1 + label smoothing 0.1          |
| A3  | A2 + Dropout 与 RandomErasing      |
| A4  | 最佳配置 + 评估期 TTA                    |
| A5  | 最佳配置的多模型集成                        |

规则：其余条件固定；每组至少 3 个随机种子；报告 PublicTest 准确率的均值与范围；不凭单次结果下结论。选定最终方案后，只对该方案在 PrivateTest 上评估一次。

## 11. 排错

| 现象                  | 可能原因                                             | 处理                                      |
| ------------------- | ------------------------------------------------ | --------------------------------------- |
| 训练准确率长期偏低（例如低于 60%） | 像素未除以 255 或归一化写错；pixels 解析后形状不对；标签错位；lr 不合适；增强过强 | 先取一个很小的子集、关闭增强，检查能否过拟合到接近 100%          |
| 训练高、验证低且差距大         | 过拟合                                              | 增强、Dropout、weight decay、label smoothing |
| 验证曲线剧烈抖动            | 学习率偏大                                            | 降低 lr，确认调度器在工作                          |
| Disgust 召回很低        | 样本极少                                             | 看混淆矩阵，用类别权重做消融                          |
| GPU 没被使用            | 装了 CPU 版 PyTorch                                 | 检查 torch.cuda.is\_available()，重装 CUDA 版 |

## 12. 需要事先核实的技术边界

- 实际 CSV 是否与公开的 FER2013 一致：列名、Usage 取值、数量、行顺序。
- 预训练权重（ImageNet 等）：属于外部数据，使用前确认是否允许；若使用，输入需放大并复制为 3 通道，且报告中注明。
- FER+ 标签：只是对同一批图像的重新标注，fer2013new.csv 与 fer2013.csv 行数、顺序相同。列顺序为 neutral、happiness、surprise、sadness、anger、disgust、fear、contempt、unknown、NF，需要重新映射到 0\~6 的编号；contempt、unknown、NF 无对应类别，需丢弃，训练集规模随之变化。无论是否使用，最终评估一律以给定的原始标签为准，PrivateTest 的 FER+ 标签不得用于训练或选模型。使用前先确认是否允许。

## 13. 参考

- WuJie1010/Facial-Expression-Recognition.Pytorch：VGG19 与 ResNet18，PrivateTest 约 73.1% 与 73.0%。
- usef-kh/fer（Khaireddin 与 Chen, 2021）：VGG 精细调参，73.28%。
- LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch：ResNet18 加训练技巧，73.70%。
- phamquiluan/ResidualMaskingNetwork（Pham 等, ICPR 2020）：单模型 74.14%，集成 76.82%。
- microsoft/FERPlus（Barsoum 等, ICMI 2016）：FER2013 的众包重标注。

以上准确率均为各项目自报，且各自的数据划分和选模型方式不尽相同，仅供量级参照。

## 14. 前端演示（本地 Web）

目标：上传图片或打开摄像头，显示 7 类表情的概率。前端只用于演示，不参与评估；对外报告的准确率一律来自 evaluate.py。项目结构中增加 app.py。

### 14.1 选型

| 方案                        | 优点                        | 缺点              | 结论         |
| ------------------------- | ------------------------- | --------------- | ---------- |
| Gradio                    | 纯 Python，几行代码得到上传、摄像头、概率条 | 界面定制有限          | 采用         |
| Streamlit                 | 同样纯 Python，适合多页面看板        | 每次交互重跑脚本，需要缓存模型 | 备选         |
| Flask 或 FastAPI + HTML/JS | 界面完全自定义                   | 前后端两套代码         | 需要自定义界面时再用 |

不做纯静态网页：静态网页无法直接运行 PyTorch 模型，要脱离 Python 需先导出 ONNX 并在浏览器端推理，复杂度超出本项目。因此前端做成本地服务，用浏览器访问终端打印的地址。

### 14.2 数据流

图像（RGB）→ 灰度 → 非 48×48 时检测并裁剪最大人脸 → 缩放到 48×48 → 与训练一致的归一化 → 模型 + 翻转 TTA → 7 类概率 → 界面。

### 14.3 技术要点

- 预处理必须与训练、评估完全一致：灰度、除以 255、同一组 MEAN 与 STD。差异不会报错，但会让准确率悄悄下降。MEAN、STD 放在公共模块（如 utils.py），训练、评估、前端共同导入。
- 人脸检测：FER2013 图像已是裁好的脸，而上传照片和摄像头画面不是。用 OpenCV 自带的 Haar 级联检测，取面积最大的人脸；未检测到时退化为整图缩放。这是本项目中唯一需要 OpenCV 的地方。
- 模型在启动时加载一次，eval 模式，推理使用 no\_grad。
- 前端不读取 PrivateTest，也不做任何选模型的操作。
- 域差异：摄像头与生活照在光照、分辨率、角度上与 FER2013 差别大，演示效果可能不如测试集，这不影响评估结论，报告里说明即可。

### 14.4 安装

```
pip install gradio opencv-python-headless
pip freeze > requirements.txt
```

### 14.5 app.py

```python
import cv2, torch
import gradio as gr
from model import build_model
from utils import MEAN, STD          # 与训练、评估共用同一组统计量

LABELS = ['愤怒', '厌恶', '恐惧', '快乐', '悲伤', '惊喜', '中立']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_model().to(dev)
model.load_state_dict(torch.load('runs/best.pt', map_location=dev)['model'])
model.eval()
face_det = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if gray.shape != (48, 48):                      # 48×48 输入跳过检测，便于对照自检
        faces = face_det.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            gray = gray[y:y + h, x:x + w]
        gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(gray).float().div(255).sub(MEAN).div(STD)[None, None].to(dev)
    with torch.no_grad():
        p = (model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)) / 2
    return {LABELS[i]: float(p[0, i]) for i in range(7)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=['upload', 'webcam'], type='numpy', label='输入图像'),
    outputs=gr.Label(num_top_classes=7, label='表情概率'),
    title='人脸表情识别',
)

if __name__ == '__main__':
    demo.launch()
```

### 14.6 运行与自检

运行 python app.py，浏览器打开终端打印的地址。自检方法：取几十张 PublicTest 的 48×48 图，分别用前端的 predict 和 evaluate.py 的流程预测，结果应一致；不一致说明预处理存在差异。Gradio 的组件参数可能随版本变化，以所装版本的官方文档为准。
