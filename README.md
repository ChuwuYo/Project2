# FER2013 人脸表情识别

用 PyTorch 从头训练的 7 类人脸表情识别（愤怒 / 厌恶 / 恐惧 / 快乐 / 悲伤 / 惊讶 / 中性）。模型是针对 48×48 灰度图改造的 ResNet18，最终用 6 个模型的概率平均集成。

## 最终结果

| 项目 | 数值 |
|---|---|
| 最终方案 | A1 与 A2 各 3 个随机种子，共 6 个 ResNet18，概率平均集成 |
| **PrivateTest 准确率** | **73.11%**（3589 张） |
| PublicTest 准确率（用于选方案） | 71.52% |
| 最佳单模型（PublicTest） | 70.30%（A1_s2） |

- PrivateTest 只在最终方案确定后评估，报告在 `reports/PrivateTest_ensemble6/`（准确率、各类 precision/recall/F1、混淆矩阵）。
- 所有选择（配置、种子、集成方式、是否翻转 TTA）只用 PublicTest，过程见 `reports/model_selection.md`。

## 数据划分（严格遵守项目要求）

| 子集 | 数量 | 用途 | 在代码中的位置 |
|---|---|---|---|
| Training | 28,709 | 更新模型参数 | `train.py` |
| PublicTest | 3,589 | 早停、选 checkpoint、比较实验、决定 TTA、选集成方案 | `train.py`、`evaluate.py --split PublicTest` |
| PrivateTest | 3,589 | 只做最终评估 | 只有 `evaluate.py` 会读取 |

`tests/test_pipeline.py` 会自动检查：训练过程从不读取 PrivateTest，且除 `evaluate.py`、`dataset.py`、`check_data.py`、`app.py` 外没有源码出现 PrivateTest。

## 实验（消融，每组只比上一组多一个改动）

| 实验 | 改动 | PublicTest 准确率 | 说明 |
|---|---|---|---|
| A0 | 基线：无数据增强 | 58.87% ± 0.61（3 个种子；翻转 TTA 59.35% ± 0.78） | 第 17 轮后严重过拟合（训练集 99.7%） |
| A1 | + 随机翻转、旋转、平移、缩放 | 69.19% ± 0.52（3 个种子；TTA 70.08% ± 0.23） | 提升约 10.6 个百分点，最明显的改进 |
| A2 | + 标签平滑 0.1 | 69.26% ± 0.46（3 个种子；TTA 69.50% ± 0.81） | 与 A1 的差距在种子波动范围内，不能说明更好 |
| A3 | + Dropout 0.4 + RandomErasing | 68.83% ± 0.31（3 个种子；TTA 69.56% ± 0.21） | 未见稳定提升，略低于 A1/A2（差距在波动范围内） |

- 结论只依据多次实验：A0→A1 的差距远大于种子间波动（标准差约 0.3–0.6%）；A1 与 A2 无显著差别，因此两者一起做集成。
- 水平翻转 TTA 在 12 个模型中的 10 个上提高了 PublicTest 准确率，是否启用由每个模型各自在 PublicTest 上决定。
- 集成：A1×3 为 71.50%，A2×3 为 70.63%，A1+A2×6 为 71.52%（均为 PublicTest），按预先约定的规则取最高者。
- 每次训练的 loss / accuracy 记录在 `runs/<实验>_s<种子>/log.csv`，曲线在同目录 `curves.png`。

训练设置：SGD（momentum 0.9，Nesterov，lr 0.01，weight decay 5e-4），batch 128，ReduceLROnPlateau（按 PublicTest 准确率，减半，patience 5），PublicTest 连续 20 轮不提升即早停，最多 100 轮，GPU 上混合精度。训练在 Colab T4 上完成，每轮约 26 秒。

## 数据检查

`python check_data.py` 生成 `reports/data_check/report.md`：三个子集数量、标签范围、图像尺寸、归一化均值/标准差全部通过。另发现 13 张几乎纯色的空白图，PublicTest 有 280 张、PrivateTest 有 288 张与 Training 像素完全相同；这是官方数据本身的情况，我们保留官方划分，不做修改。

## 目录

```
dataset.py        数据读取（CSV→缓存）、划分、预处理与增强（训练/评估/前端共用）
model.py          改造的 ResNet18
utils.py          随机种子、DataLoader、推理（含翻转 TTA）、模型加载、画图
train.py          训练（A0–A3），断点续训
evaluate.py       评估；唯一读取 PrivateTest 的脚本；支持多模型集成
check_data.py     数据检查
app.py            Gradio 网页演示（实时摄像头 / 上传拍照）
final_runs.txt    最终模型列表（evaluate.py 和 app.py 默认读取）
runs/             每次训练的权重、日志、曲线（压缩包只保留最终 6 个模型的 best.pt）
reports/          数据检查、方案选择、PrivateTest 报告
tests/            自动测试
assets/           YuNet 人脸检测模型（演示用）
docs/             项目要求 PDF、技术方案、项目代码详解
scripts/           Windows 一键脚本：start_demo.bat、verify_test.bat、setup_env.bat
```

## 使用（Windows）

1. 把数据文件 `icml_face_data.csv` 放进 `data/`（压缩包不含数据）。
2. 双击 `scripts\verify_test.bat`：在 PrivateTest 上复核准确率，应输出 0.7311。
3. 双击 `scripts\start_demo.bat`：启动网页演示并自动打开浏览器。

首次运行会自动创建 `.venv` 并安装依赖；检测到 NVIDIA 显卡时自动安装 GPU 版 PyTorch，没有显卡也能用 CPU 运行。第一次读取数据会解析 CSV 并生成缓存 `data/fer2013.npz`（约 1 分钟），之后约 1 秒。

命令行：
```
pip install -r requirements.txt
# 有 NVIDIA 显卡时改装 GPU 版 PyTorch：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python check_data.py                    # 数据检查
python train.py --exp A1 --seed 0       # 训练一个配置（--resume 断点续训）
python evaluate.py --split PublicTest   # 在验证集上比较
python evaluate.py                      # 最终评估：final_runs.txt 中的模型，PrivateTest
python app.py                           # 网页演示
pytest tests                            # 自动测试
```

## 网页演示

- “实时摄像头”：每帧检测人脸、画框并标出表情（可同时多张脸）。
- “上传 / 拍照”：上传、粘贴或拍照后点“识别表情”，显示 7 类概率和模型实际看到的 48×48 人脸。
- 人脸检测用 OpenCV 自带的 YuNet，按双眼转正后裁剪。演示只用于展示，不参与任何评估。
- 模型只在 FER2013 上训练，73.11% 是该测试集上的准确率；真实摄像头的光线、角度和人脸风格不同，表情夸张一些更容易识别对。

## 局限

- A3 有两个种子在 100 轮上限附近才达到最佳（第 92、81 轮），可能没完全收敛。
- 没有系统地比较学习率、批次大小和其他网络结构。
- 厌恶（Disgust）训练样本只有 436 张，恐惧（Fear）的召回率最低（50.6%），与悲伤、愤怒容易混淆（见混淆矩阵）。

## 参考与致谢

- [LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch](https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch)：参考了“ResNet 首层改为 3×3、去掉 maxpool”的结构思路，以及数据增强、标签平滑、测试时增强等训练技巧（它用 TenCrop，我们用更简单的水平翻转）。代码为我们自己编写，未使用其权重和结果；该仓库把 PrivateTest 当验证集选模型，我们改为只用 PublicTest。地址记录在 `references/references.yaml`。
- [YuNet 人脸检测](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)（OpenCV Zoo，MIT 许可）：仅用于网页演示中的人脸定位。
- 框架：PyTorch、torchvision、scikit-learn、matplotlib、Gradio、OpenCV。
