# 方案选择（只用 PublicTest，未读取 PrivateTest）

## 各 run 的 PublicTest 准确率
| run | 最佳轮 | val_acc | val_acc（翻转 TTA） | 采用 TTA |
|---|---|---|---|---|
| A0_s0（无增强） | 17 | 0.5862 | 0.5857 | 否 |
| A1_s0（+增强） | 50 | 0.6918 | 0.7008 | 是 |
| A1_s1 | 55 | 0.6868 | 0.6985 | 是 |
| A1_s2 | 77 | 0.6971 | 0.7030 | 是 |
| A2_s0（+标签平滑0.1） | 80 | 0.6932 | 0.6991 | 是 |
| A2_s1 | 67 | 0.6877 | 0.6857 | 否 |
| A2_s2 | 54 | 0.6969 | 0.7002 | 是 |
| A3_s0（+Dropout0.4+RandomErasing） | 92 | 0.6857 | 0.6943 | 是 |

阶段 1 按 PublicTest 选出前 2 名配置：A1、A2，各补 seed1、seed2。

## 候选方案（evaluate.py --split PublicTest，3589 张）
| 方案 | PublicTest 准确率 |
|---|---|
| 最佳单模型 A1_s2 | 0.7030 |
| A1 三个 seed 集成 | 0.7150 |
| A2 三个 seed 集成 | 0.7063 |
| A1+A2 六模型集成 | **0.7152** |

选定：A1+A2 六模型集成（PublicTest 最高；与 A1 三模型集成仅差 1 张图，按预先约定的规则取最高者）。写入 final_runs.txt。
明细在 reports/selection/。
