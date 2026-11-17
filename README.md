# FER2013 人脸表情识别

用 PyTorch 从头训练的 7 类人脸表情识别（愤怒 / 厌恶 / 恐惧 / 快乐 / 悲伤 / 惊讶 / 中性）。模型是针对 48×48 灰度图改造的 ResNet18，最终用 9 个模型（A1、A2、A5 各 3 个种子）的概率平均集成。

## 最终结果

| 项目 | 数值 |
|---|---|
| 最终方案 | A1、A2、A5 各 3 个随机种子，共 9 个 ResNet18，概率平均集成 |
| **PrivateTest 准确率** | **73.50%**（3589 张） |
| PublicTest 准确率（用于选方案） | 71.69% |
| 最佳单模型（PublicTest） | 71.05%（hp_bs64/A1_s0：A1 配置 + batch 64；PrivateTest 71.97%） |
| 上一版最终方案（6 个模型，A1+A2） | PublicTest 71.52%，PrivateTest 73.11%（报告保留在 `reports/PrivateTest_ensemble6/`） |

- PrivateTest 只在最终方案确定后评估，报告在 `reports/PrivateTest_ensemble6/`（准确率、各类 precision/recall/F1、混淆矩阵）。
- 所有选择（配置、种子、集成方式、是否翻转 TTA）只用 PublicTest，过程见 `reports/model_selection.md`。

## 数据划分（严格遵守项目要求）

| 子集 | 数量 | 用途 | 在代码中的位置 |
|---|---|---|---|
| Training | 28,709 | 更新模型参数 | `train.py` |
| PublicTest | 3,589 | 早停、选 checkpoint、比较实验、决定 TTA、选集成方案 | `train.py`、`evaluate.py --split PublicTest` |
| PrivateTest | 3,589 | 只做最终评估 | 只有 `evaluate.py` 会读取 |

`tests/test_pipeline.py`（9 项）会自动检查：训练过程从不读取 PrivateTest，且除 `evaluate.py`、`dataset.py`、`check_data.py`、`app.py` 外没有源码出现 PrivateTest。

## 实验（消融，每组只比上一组多一个改动）

| 实验 | 改动 | PublicTest 准确率 | 说明 |
|---|---|---|---|
| A0 | 基线：无数据增强 | 58.87% ± 0.61（3 个种子；翻转 TTA 59.35% ± 0.78） | 第 17 轮后严重过拟合（训练集 99.7%） |
| A1 | + 随机翻转、旋转、平移、缩放 | 69.19% ± 0.52（3 个种子；TTA 70.08% ± 0.23） | 提升约 10.3 个百分点，最明显的改进 |
| A2 | + 标签平滑 0.1 | 69.26% ± 0.46（3 个种子；TTA 69.50% ± 0.81） | 与 A1 的差距在种子波动范围内，不能说明更好 |
| A3 | + Dropout 0.4 + RandomErasing | 68.83% ± 0.31（3 个种子；TTA 69.56% ± 0.21） | 未见稳定提升，略低于 A1/A2（差距在波动范围内） |
| A5 | A1 + 类别加权损失（权重 = 总数 / (7 × 该类数量)） | 68.88% ± 0.30（3 个种子；TTA 69.44% ± 0.55） | 低于 A1（70.08%），没有帮助；准确率上看不出改善少数类 |
| V1 | A1 的增强，模型换成 VGG 风格 CNN（8 个卷积层，无残差） | 68.71% ± 0.58（3 个种子；TTA 69.10% ± 0.31） | 比 ResNet18（A1）低约 1 个百分点 |

- 结论只依据多次实验：A0→A1 的差距远大于种子间波动（标准差约 0.3–0.6%）；A1 与 A2 无显著差别，因此两者一起做集成。
- 水平翻转 TTA 在 12 个模型中的 10 个上提高了 PublicTest 准确率，是否启用由每个模型各自在 PublicTest 上决定。
- 集成（均为 PublicTest）：A1×3 为 71.50%，A2×3 为 70.63%，A1+A2×6 为 71.52%，A5×3 为 70.94%，A1+A2+A5×9 为 71.69%。按预先约定的规则（集成超过 71.52% 才换），最终方案换成 9 模型集成；但它只比 6 模型多对 6 张图，属于很小的差别。
- 超参数（只做 A1 种子 0，每项 1 次，不做均值）：lr 0.005 为 68.60%；lr 0.02 最佳轮 69.71%（不带 TTA，权重丢失，见局限）；batch 64 为 71.05%；batch 256 为 68.40%；基准 A1_s0（lr 0.01，batch 128）为 70.08%。batch 64 最好，但只有 1 个种子，差距（+0.97）可能有一部分是随机波动。
- 每次训练的 loss / accuracy 记录在 `runs/<实验>_s<种子>/log.csv`，曲线在同目录 `curves.png`。

训练设置：SGD（momentum 0.9，Nesterov，lr 0.01，weight decay 5e-4），batch 128，ReduceLROnPlateau（按 PublicTest 准确率，减半，patience 5），PublicTest 连续 20 轮不提升即早停，最多 100 轮，GPU 上混合精度。训练在 Colab T4 上完成，每轮约 26 秒；具体怎么训练、怎么用 Colab、怎么复现见 `docs/训练过程说明.md`。

## 数据检查

`python check_data.py` 生成 `reports/data_check/report.md`：三个子集数量、标签范围、图像尺寸、归一化均值/标准差全部通过。另发现 13 张几乎纯色的空白图，PublicTest 有 280 张、PrivateTest 有 288 张与 Training 像素完全相同；这是官方数据本身的情况，我们保留官方划分，不做修改。

## 目录

```
dataset.py        数据读取（CSV→缓存）、划分、预处理与增强（训练/评估/前端共用）
model.py          改造的 ResNet18
utils.py          随机种子、DataLoader、推理（含翻转 TTA）、模型加载、画图
train.py          训练（A0–A3、A5、V1），断点续训
evaluate.py       评估；唯一读取 PrivateTest 的脚本；支持多模型集成
check_data.py     数据检查
app.py            Gradio 网页演示（实时摄像头 / 上传拍照）
runs/final_runs.txt    最终模型列表（9 个，多模型平均；evaluate.py 和 app.py 默认读取）
runs/best_single.txt   最佳单模型（hp_bs64/A1_s0；evaluate.py / app.py 加 --single 时读取）
runs/             每次训练的权重、日志、曲线（压缩包不带 A0、A3、V1、其余 hp 实验的权重；最终 9 个模型和最佳单模型的 best.pt 在各自的交付包里）
reports/          数据检查、方案选择、PrivateTest 报告
tests/            自动测试
assets/           YuNet 人脸检测模型（演示用）
docs/             项目要求 PDF、技术方案、项目代码详解、训练过程说明
scripts/          Windows 一键脚本：verify_test.bat（复核准确率）、start_demo.bat（演示）、train.bat（训练）、setup_env.bat（装环境）
```

## 使用（Windows）

1. 把数据文件 `icml_face_data.csv`（或 `fer2013.csv`）放进 `data/`（压缩包不含数据）。
2. 双击 `scripts\verify_test.bat`，输 1（多模型平均）：用交付的 9 个模型在 PrivateTest 上复核准确率，应输出 0.7350；输 2（最佳单模型 hp_bs64/A1_s0）应输出 0.7197。
3. 双击 `scripts\start_demo.bat`：同样先选 1 或 2，然后启动网页演示并自动打开浏览器。
4. 想自己重新训练一个模型：双击 `scripts\train.bat`（默认 A1、种子 0，结果在 `runs_local\A1_s0\`；中断后再双击会接着训练）。详细步骤、怎么看结果见 `docs/训练过程说明.md` 第 7 节。

首次运行会自动创建 `.venv` 并安装依赖；检测到 NVIDIA 显卡时自动安装 GPU 版 PyTorch，没有显卡也能用 CPU 运行（评估和演示可以，训练会非常慢）。第一次读取数据会解析 CSV 并生成缓存 `data/fer2013.npz`（约 1 分钟），之后约 1 秒。

命令行：
```
pip install -r requirements.txt
# 有 NVIDIA 显卡时改装 GPU 版 PyTorch：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python check_data.py                                        # 数据检查
python evaluate.py                                          # 复核最终结果：runs/final_runs.txt 中的 9 个模型，PrivateTest
python evaluate.py --single                                 # 最佳单模型 runs/best_single.txt（hp_bs64/A1_s0），PrivateTest
python train.py --exp A1 --seed 0 --out runs_local          # 自己训练一个模型（中断后加 --resume）
python evaluate.py --runs runs_local/A1_s0 --split PublicTest   # 在验证集上看自己训练的模型
python app.py                                               # 网页演示（最终模型）
python app.py --runs runs_local/A1_s0                       # 网页演示（自己训练的模型）
pytest tests                                                # 自动测试（9 项）
```

`runs/` 里是交付的最终模型，自己训练请始终加 `--out runs_local`。即使忘了加，`train.py` 发现目录里已有模型（`best.pt`）也会报错停下，不会覆盖；`evaluate.py` 也不会用另一组模型的结果覆盖 `reports/` 里的最终 PrivateTest 报告。

## 网页演示

- “实时摄像头”：每帧检测人脸、画框并标出表情（可同时多张脸）。
- “上传 / 拍照”：上传、粘贴或拍照后点“识别表情”，显示 7 类概率和模型实际看到的 48×48 人脸。
- 人脸检测用 OpenCV 自带的 YuNet，按双眼转正后裁剪。演示只用于展示，不参与任何评估。
- 模型只在 FER2013 上训练，73.50% 是该测试集上的准确率；真实摄像头的光线、角度和人脸风格不同，表情夸张一些更容易识别对。

## 局限

- A3 有两个种子在 100 轮上限附近才达到最佳（第 92、81 轮），可能没完全收敛。
- 学习率、批次大小每项只试了 1 个种子；hp_lr0.02 续训时旧 VM 上的 best.pt 丢了，只有 log.csv 里的 69.71%（无 TTA），没有权重和 TTA 结果。
- 类别加权（A5）和 VGG（V1）已试，没有超过 A1 单模型，A5 加入集成后 PublicTest 从 71.52% 到 71.69%（只多对 6 张图）。
- 厌恶（Disgust）训练样本只有 436 张，恐惧（Fear）的召回率最低（51.1%），与悲伤、愤怒容易混淆（见混淆矩阵）。

## 参考与致谢

- [LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch](https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch)：参考了“ResNet 首层改为 3×3、去掉 maxpool”的结构思路，以及数据增强、标签平滑、测试时增强等训练技巧（它用 TenCrop，我们用更简单的水平翻转）。代码为我们自己编写，未使用其权重和结果；该仓库把 PrivateTest 当验证集选模型，我们改为只用 PublicTest。地址记录在 `references/references.yaml`。
- [YuNet 人脸检测](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)（OpenCV Zoo，MIT 许可）：仅用于网页演示中的人脸定位。
- 框架：PyTorch、torchvision、scikit-learn、matplotlib、Gradio、OpenCV。
