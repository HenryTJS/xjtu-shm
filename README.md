# 疲劳机多源在线损伤度 D(t) 监测 —— 总纲与使用说明

## 1. 项目简介与正式范围

- **数据**：疲劳机多源监测，每试件含 光纤(FO)/声发射(AE)/应变(strain) 三路，10Hz 统一时间轴。
- **正式主样本 = 6 组：016、017、018、019、020、022**。
- **目标**：由多源信号在线算出连续损伤度 $D(t)\in[0,1]$，按 0.25/0.55/0.85 分级预警（注意/预警/临危），供论文与专利使用。
- **客观失效锚 b3 = 数据末端(≈99% 寿命) = 断裂时刻**；b2(扩展) 仅作离线评估/选参参考；b1(萌生) 不用于评估。

## 2. 目录结构

```
d:\lixiang\
├── README.md               # 本文件
├── 数据记录.xlsx           # 实验载荷程序/采集信息
├── 001..027\               # 原始试件数据(光纤/AE/应变 CSV)
├── aligned\                # 多源对齐输出 + 对齐元信息
├── shm\                    # 库包
│   ├── config.py           # 全局常量
│   ├── data_loader.py      # DataLoader：多源加载
│   ├── streaming.py        # ChunkedDataReader + StreamSimulator
│   └── damage_index.py     # OnlineDamageIndex：连续损伤度 D(t) + 分级预警
├── prepare_data.py         # 阶段① 数据准备: align / weaklabels
├── evaluate.py             # 阶段② 评估与出图: degree / warning / curves / paper
├── robustness.py           # 阶段③ 稳健性: sens / loso / ablation / stats
├── eval_common.py          # 公共库
├── results\                # 生成的 CSV
├── figures\                # 生成的图
├── cache\                  # 逐点 D 缓存
└── weak_labels\            # 弱标签生成结果(6 组 *_label.csv + labels_summary.csv)
```

## 3. 正式方法：连续损伤度 D(t)

**证据层**：

- `e_dmg`：AE"损伤型事件"能量（logE 超滚动背景分位+抬升量即判定）→ 块能量 log EWMA；
- `e_full`：AE 全能量累积 log 加速度（短/长窗斜率差 EWMA）；
- `e_ae = max(e_dmg, e_full)`；
- `e_strain`：应变块 std 相对运行中位发散（辅证，降权）。

**状态层**：`risk = max(e_ae, e_strain)` → D 升快降慢累积 → 分级 0.25/0.55/0.85。
全程**因果在线流式**（逐点，零未来信息）；D 计算不含标签。
可调参数集中在构造参数（rise/fall/acc_scale/estrain_w/lift 等）；含消融开关(abl)与结构试验开关。

**损伤确认后加速追赶(latch, 2026-09-08 起默认开)**：D 慢 rise 追不满断裂前脉冲证据曾使 0.85(临危)级
0/6 不可达。现当 D 经**不可逆确认**(最近 3 块 min≥0.15 且 ≥0.30，与 A-预警判据同源)后，对 risk 用快 rise
追赶 → 达 0.85 级 0/6→6/6，且预警 onset/分级完全不变(无副作用，6 组验证)。

**评估口径**：A-预警 = D 首次"不可逆"≥0.3（2% 寿命窗内不回落到 0.15）；与 b2 参考对齐、b3 客观锚。

## 4. 弱标签体系

- 协议：b3=末端断裂(≈99%)；b2=扩展(AE 累积 log 最大加速 或 应变 std 发散更早)；b1=萌生(主样本各试件 b1 收敛于 b2，无独立 AE 萌生段)。
- 生成：`prepare_data.py weaklabels` → `weak_labels/{gid}_label.csv` + `labels_summary.csv`。

| gid | b2(扩展) | b3(断裂) |
| --- | -------- | -------- |
| 016 | 43.8     | 99       |
| 017 | 82.5     | 99       |
| 018 | 69.7     | 99       |
| 019 | 77.0     | 99       |
| 020 | 70.2     | 99       |
| 022 | 77.9     | 99       |

## 5. 主样本 6 组结果

| gid | D_end | 达0.85(t85%) | 断裂前单调(尾段上升占比) | A-预警 t_warn(%) | b2(%) | 相对 b2 偏差 | 断裂前提前 lead(%) |
| --- | ----- | ------------ | ------------------------ | ---------------- | ----- | ------------ | ------------------ |
| 016 | 0.94  | 95.8         | ≥99.8%                  | 46.9             | 43.8  | +3.1         | 52.1               |
| 017 | 0.96  | 87.0         | ≥99.8%                  | 85.9             | 82.5  | +3.4         | 13.1               |
| 018 | 0.94  | 73.8         | ≥99.8%                  | 72.4             | 69.7  | +2.7         | 26.6               |
| 019 | 0.95  | 81.3         | ≥99.8%                  | 80.0             | 77.0  | +3.0         | 19.0               |
| 020 | 0.87  | 74.7         | ≥99.8%                  | 73.0             | 70.2  | +2.8         | 26.0               |
| 022 | 1.00  | 96.0         | ≥99.8%                  | 86.1             | 77.9  | +8.2         | 12.9               |

**结论**：预警 **6/6 全部精准**(对齐扩展 b2，偏差 +2.7-+8.2)；断裂前给出 12.9-52.1% 提前量；D 断裂前单调不回落；**达 0.55(预警)级 6/6、达 0.85(临危)级 6/6**（latch，t85 断裂前 73.8~96.0%）。方法结构经参数敏感性(±30%)、留一交叉验证与证据消融检验稳健。

## 6. 统计检验

脚本 `robustness.py stats` → `results/statistical_test.csv`。与单源基线（仅AE=`no_strain`、仅应变=`only_strain`，同一 A-预警判据）配对比较，指标 = \|预警点−b2\|。

| gid | \|D\| | \|仅AE\| | \|仅应变\| |
| --- | ----- | -------- | ---------- |
| 016 | 3.1   | 3.1      | 48.0       |
| 017 | 3.4   | 3.4      | 14.2       |
| 018 | 2.7   | 2.7      | 漏报       |
| 019 | 3.0   | 3.0      | 漏报       |
| 020 | 2.8   | 2.8      | 漏报       |
| 022 | 8.2   | 19.0     | 8.2        |

- **单源应变漏报 3/6**（018/019/020 应变无不可逆发散），仅AE 漏报 0 → 多源融合相对单源应变的优势为结构性（应变对静默型失效）。
- **D 自身精度（Bootstrap 试件重采样 95% CI）**：\|err\| 均值 CI [2.9, 5.6]（样本均值 3.9）；断裂前提前量 lead 均值 CI [16.2, 36.7]（均值 25.0）。
- **配对 Wilcoxon**：D vs 仅AE p=0.50、D vs 仅应变 p=0.25 —— 均不显著。原因：主样本 6 组中 5 组本由 AE 证据主导（D 与仅AE 几乎同值），仅 022 体现融合增益；且 n=6 功效极低。**如实注明：小样本 + 主样本 AE 主导下，配对检验无法给出显著差异，结论以 Bootstrap CI 与结构性漏报差异为主。**

## 7. 脚本说明

| 脚本                | 子命令         | 作用                                                                          |
| ------------------- | -------------- | ----------------------------------------------------------------------------- |
| `prepare_data.py` | `align`      | 多源数据对齐(AE 网格)→`aligned/`                                           |
|                     | `weaklabels` | 弱标签生成 →`weak_labels/`                                                 |
| `evaluate.py`     | `degree`     | D 达阈/单调 →`results/damage_degree_metrics.csv`、缓存 `cache/_hi_cache` |
|                     | `warning`    | A-预警 onset 与分级 →`results/warning_onset.csv`(读缓存,先跑 degree)       |
|                     | `curves`     | 6 组 D(t) 曲线 →`figures/damage_degree_curves.png`                         |
|                     | `paper`      | 论文四联图+流程图 →`figures/paper_<gid>.png`、`method_flowchart.png`     |
| `robustness.py`   | `sens`       | 参数 ±30% 敏感性 →`results/parameter_sensitivity.csv`、缓存 `_t7_cache` |
|                     | `loso`       | 留一试件 CV 选参 →`results/leave_one_out_cv.csv`(需先 sens)                |
|                     | `ablation`   | 证据消融 →`results/ablation.csv`、`figures/ablation.png`                 |
|                     | `stats`      | T9 统计(配对+Bootstrap) →`results/statistical_test.csv`                    |
| `eval_common.py`  | —             | 公共库(组集/指标/缓存/流式)                                                   |

> 说明：早期同名独立脚本已全部合并进上表 3 个阶段脚本，勿再引用旧文件名。

## 8. 运行与复现

```bash
# 环境(conda xjtushm)
C:\Users\ASUS\.conda\envs\xjtushm\python.exe
# 阶段① 数据准备：弱标签 + 多源对齐(6 组主样本)
python prepare_data.py weaklabels align
# 阶段② 主样本评估与出图(6 组)：D 达阈 → A-预警 → 曲线/论文图
#   (重算 D 需先清缓存) Remove-Item -Recurse -Force cache\_hi_cache
python evaluate.py degree warning curves   # --workers 8
python evaluate.py paper
# 阶段③ 稳健性(敏感性→留一CV 需前者缓存；消融/统计独立)
python robustness.py sens --workers 8
python robustness.py loso ablation stats
# 等价写法: all 一次跑完该阶段全部任务
python evaluate.py all
```

注意：多进程脚本在 `d:\lixiang` 下运行；AE CSV `encoding='utf-8-sig'`；缓存按参数指纹隔离。

## 9. 待办 / 问题 / 决策点

- **待数据方核实**：其余试件 AE 采集/处理口径，核实后决定是否纳入正式结论。
- **未做**：T11 与深度方法对照、T12 实时化延迟、T13 专利交底书。

## 10. 数据与处理说明

- **采样**：光纤/应变 10Hz 连续记录；AE 为事件/窗口特征(每 0.1s 窗口整合出 25 指标，密度因试件而异)。
- **对齐**：`prepare_data.py align` 以 AE 事件为网格、应变时间为主基准；`DataLoader/ChunkedDataReader` 逐点流式读取，AE 事件门控(仅新事件计入能量，避免稀疏组能量重复计数)。
- **分块**：`EXT_BLOCK_PTS=500`(约 50s) 为 D 块级统计单位。
