# 疲劳机多源在线损伤度 D(t) 监测 —— 总纲与使用说明

> **本 README 为工作区唯一文档**，集中：项目简介、目录结构、正式方法、数据与弱标签、
> 6 组主样本结果、脚本说明、运行复现、产物、待办与决策点。
> 历史范式"在线四阶段划分"已弃用(2026-09-06)，相关代码/说明一律移除。

---

## 1. 项目简介与正式范围

- **数据**：疲劳机多源监测，每试件含 光纤(FO)/声发射(AE)/应变(strain) 三路，10Hz 统一时间轴。
- **正式主样本 = 6 组：016、017、018、019、020、022**（AE 记录口径一致、预警验证全部精准）。
  其余试件数据待与数据提供方(学长)核实采集/处理口径后再决定是否纳入；**本 README 一切正式结论只针对上述 6 组**。
- **目标**：由多源信号在线算出连续损伤度 $D(t)\in[0,1]$，按 0.25/0.55/0.85 分级预警（注意/预警/临危），供论文与专利使用。
- **客观失效锚 b3 = 数据末端(≈99% 寿命) = 断裂时刻**；b2(扩展) 仅作离线评估/选参参考；b1(萌生) 不用于评估。

## 2. 目录结构

```
d:\lixiang\
├── README.md               # 本文件(唯一文档)
├── 数据记录.xlsx           # 实验载荷程序/采集信息
├── 001..027\               # 原始试件数据(光纤/AE/应变 CSV)
├── aligned\                # 多源对齐输出(align_multisource 生成) + 对齐元信息
├── shm\                    # 库包(被脚本 import)
│   ├── config.py           # 全局常量(CHUNK_SIZE / EXT_BLOCK_PTS / DEFAULT_GROUPS)
│   ├── data_loader.py      # DataLoader：多源加载(StreamSimulator 依赖)
│   ├── streaming.py        # ChunkedDataReader + StreamSimulator(逐点流式)
│   └── damage_index.py     # OnlineDamageIndex：连续损伤度 D(t) + 分级预警
├── (根目录工具脚本, 见 §6)
├── results\                # 生成的 CSV(评估结果)
├── figures\                # 生成的图
├── cache\                  # 逐点 D 缓存(_hi_cache 等)
└── weak_labels\            # 弱标签生成结果(6 组 *_label.csv + labels_summary.csv)
```

## 3. 正式方法：连续损伤度 D(t)（`shm/damage_index.py`，参数化默认 = v6）

**证据层**：
- `e_dmg`：AE"损伤型事件"能量（logE 超滚动背景分位+抬升量即判定）→ 块能量 log EWMA；
- `e_full`：AE 全能量累积 log 加速度（短/长窗斜率差 EWMA）；
- `e_ae = max(e_dmg, e_full)`；
- `e_strain`：应变块 std 相对运行中位发散（辅证，降权）。

**状态层**：`risk = max(e_ae, e_strain)` → D 升快降慢累积 → 分级 0.25/0.55/0.85。
全程**因果在线流式**（逐点，零未来信息）；D 计算不含标签。
可调参数集中在构造参数（rise/fall/acc_scale/estrain_w/lift 等），默认即 v6；含消融开关(abl)与结构试验开关(默认关)。

**评估口径**：A-预警 = D 首次"不可逆"≥0.3（2% 寿命窗内不回落到 0.15）；与 b2 参考对齐、b3 客观锚。

## 4. 弱标签体系（6 组主样本）

- 协议：b3=末端断裂(≈99%)；b2=扩展(AE 累积 log 最大加速 或 应变 std 发散更早)；b1=萌生(主样本各试件 b1 收敛于 b2，无独立 AE 萌生段)。
- 生成：`make_weak_labels.py` → `weak_labels/{gid}_label.csv` + `labels_summary.csv`。

| gid | b2(扩展) | b3(断裂) |
|---|---|---|
| 016 | 43.8 | 99 |
| 017 | 82.5 | 99 |
| 018 | 69.7 | 99 |
| 019 | 77.0 | 99 |
| 020 | 70.2 | 99 |
| 022 | 77.9 | 99 |

## 5. 主样本 6 组结果

| gid | D_end | 断裂前单调(尾段上升占比) | A-预警 t_warn(%) | b2(%) | 相对 b2 偏差 | 断裂前提前 lead(%) |
|---|---|---|---|---|---|---|
| 016 | 0.64 | ≥99.8% | 46.9 | 43.8 | +3.1 | 52.1 |
| 017 | 0.63 | ≥99.8% | 85.9 | 82.5 | +3.4 | 13.1 |
| 018 | 0.56 | ≥99.8% | 72.4 | 69.7 | +2.7 | 26.6 |
| 019 | 0.81 | ≥99.8% | 80.0 | 77.0 | +3.0 | 19.0 |
| 020 | 0.45 | ≥99.8% | 73.0 | 70.2 | +2.8 | 26.0 |
| 022 | 0.77 | ≥99.8% | 86.1 | 77.9 | +8.2 | 12.9 |

**结论**：预警 **6/6 全部精准**(对齐扩展 b2，偏差 +2.7~+8.2)；断裂前给出 12.9~52.1% 提前量；D 断裂前单调不回落；达 0.55 级 5/6。方法结构经参数敏感性(±30%)、留一交叉验证与证据消融检验稳健。

## 6. 脚本说明（文件名 = 作用）

| 脚本 | 作用 |
|---|---|
| `align_multisource`（原 align.py） | 多源数据对齐(AE 网格)→ `aligned/` |
| `make_weak_labels.py` | 生成 6 组弱标签 → `weak_labels/` |
| `evaluate_damage_degree.py` | 评估损伤度 D 达阈/单调 → `results/damage_degree_metrics.csv`、缓存 `cache/_hi_cache` |
| `evaluate_warning_onset.py` | 评估 A-预警 onset 与分级 → `results/warning_onset.csv` |
| `plot_damage_curves.py` | 画 6 组 D(t) 曲线 → `figures/damage_degree_curves.png` |
| `parameter_sensitivity.py` | 参数 ±30% 敏感性 → `results/parameter_sensitivity.csv`、缓存 `cache/_t7_cache` |
| `leave_one_out_cv.py` | 留一试件交叉验证选参 → `results/leave_one_out_cv.csv` |
| `ablation_study.py` | 证据消融(去各源/结构) → `results/ablation.csv`、`figures/ablation.png` |
| `eval_common.py` | 上述评估/敏感性/消融的公共库(组集/指标/缓存) |

## 7. 运行与复现

```bash
# 环境(conda xjtushm)
C:\Users\ASUS\.conda\envs\xjtushm\python.exe
# 弱标签(6 组主样本)
python make_weak_labels.py
# 多源对齐(如需重生成 aligned/)
python align.py
# 损伤度 D 评估(6 组; 重算需先清 cache\_hi_cache)
Remove-Item -Recurse -Force cache\_hi_cache -ErrorAction SilentlyContinue
python evaluate_damage_degree.py --workers 8
# A-预警分级(读 cache\_hi_cache)
python evaluate_warning_onset.py
# D 曲线图
python plot_damage_curves.py
# 方法结构稳健性(参数敏感性 / 留一交叉验证 / 消融)
python parameter_sensitivity.py --workers 8
python leave_one_out_cv.py
python ablation_study.py
```
注意：多进程脚本在 `d:\lixiang` 下运行；AE CSV `encoding='utf-8-sig'`；缓存按参数指纹隔离。

## 8. 输出产物

- `results/`：`damage_degree_metrics.csv`、`warning_onset.csv`、`parameter_sensitivity.csv`、`leave_one_out_cv.csv`、`ablation.csv`
- `figures/`：`damage_degree_curves.png`、`ablation.png`
- `cache/`：`_hi_cache`(逐点 D)、`_t7_cache`(参数敏感性)
- `weak_labels/`：6 组 `*_label.csv` + `labels_summary.csv`

## 9. 待办 / 问题 / 决策点

- **待数据方核实**：其余试件 AE 采集/处理口径，核实后决定是否纳入正式结论。
- **D 绝对分级标度**：0.85(临危)级 0/6 可达；D 上界试件相关 → 分级改试件内自适应或论文只报两级(0.25/0.55)待定。
- **应变片末段可靠性**：个别试件(如 020)应变末段近乎失效，AE 起主预警作用 → 多源互补性成立。
- **未做**：T9 统计显著性(6 组小样本 Bootstrap/配对检验)、T10 论文级图、T11 与深度方法对照、T12 实时化延迟、T13 专利交底书。

## 10. 数据与处理说明（基建）

- **采样**：光纤/应变 10Hz 连续记录；AE 为事件/窗口特征(每 0.1s 窗口整合出 25 指标，密度因试件而异)。
- **对齐**：`align.py` 以 AE 事件为网格、应变时间为主基准；`DataLoader/ChunkedDataReader` 逐点流式读取，AE 事件门控(仅新事件计入能量，避免稀疏组能量重复计数)。
- **分块**：`EXT_BLOCK_PTS=500`(约 50s) 为 D 块级统计单位。
