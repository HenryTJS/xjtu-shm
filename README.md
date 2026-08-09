# 在线流式阶段划分方案（001-027）

## 1. 设计目标

将当前离线批处理架构改造为**在线流式处理架构**，满足以下要求：

- **实时性**：每个数据点到达后立即处理，不等待未来数据
- **无先验知识**：不依赖全局统计量（百分位、均值、标准差等）
- **数据驱动**：不依赖工况记录，完全基于传感器数据
- **可视化**：实时模拟数据流输入，动态展示阶段划分结果
- **范围**：支持所有组号（001-027），通过命令行参数 `--group` / `--groups` 指定

## 2. 数据概况（001-027）

| 组号    | 应变数据                | 声发射(AE)数据           | 光纤(FO)数据            | 数据量级 | 关键特征                                                                                  |
| ------- | ----------------------- | ------------------------ | ----------------------- | -------- | ----------------------------------------------------------------------------------------- |
| 001-015 | 各组的应变.csv          | 各组的声发射.csv         | 各组的光纤.csv          | ~30k-50k | 部分组缺少光纤或声发射数据                                                                |
| 016     | 016应变.csv (50,120行)  | 016声发射.csv (31,444行) | 016光纤.csv (50,228行)  | ~50k     | 应变均值7.6，最后段跳升至21.7；AE有474个Kurt>100事件；光纤5通道平稳                       |
| 017     | 017应变.csv (46,003行)  | 017声发射.csv (46,568行) | 017光纤.csv (46,003行)  | ~46k     | 应变呈循环加载模式，最后段跳升至60.9；AE有989个Kurt>100事件；光纤s4通道剧烈交替振荡       |
| 018     | 018应变.csv (35,216行)  | 018声发射.csv (35,225行) | 018光纤.xlsx            | ~35k     | 应变均值18.4，最后段升至28.2；AE有493个Kurt>100事件；光纤仅1通道(Fiber_s1)                |
| 019     | 019应变.xlsx (39,989行) | 019声发射.csv (40,001行) | 019光纤.xlsx            | ~40k     | **关键故障组**：应变在t≈106s从0.1跳变至8.4，AE出现Kurtosis>1000的极端事件(均值379) |
| 020     | 020应变.xlsx (58,893行) | 020声发射.csv (58,904行) | 020光纤.xlsx (58,893行) | ~59k     | 应变均值36.5，前段26.9后段38.8；AE仅247个Kurt>100事件；光纤5通道大幅波动(-71~+47)         |
| 021-027 | 各组的应变.csv/xlsx     | 各组的声发射.csv         | 各组的光纤.csv/xlsx     | ~30k-60k | 部分组缺少光纤数据                                                                        |

> **注意**：并非所有组都有完整的三类传感器数据。系统在运行时自动检测可用传感器，缺失的传感器通道会被跳过，不影响整体处理流程。

## 3. 当前离线架构的离线依赖分析

| 模块                                         | 离线方法                                      | 问题                   | 在线替代方案                |
| -------------------------------------------- | --------------------------------------------- | ---------------------- | --------------------------- |
| `StageDivider._detect_strain_jump`         | `np.percentile(sd, 99.5)` 全局百分位        | 需要全部数据才能计算   | 滑动窗口百分位 + 自适应阈值 |
| `StageDivider.detect_strain_change_points` | `rpt.Binseg` 变点检测                       | 需要完整序列，无法增量 | 状态机 + 局部跳变检测       |
| `StageDivider.detect_ae_stages`            | `rpt.Binseg` + `np.percentile(ce, 33/66)` | 需要完整累积能量曲线   | 累积能量斜率变化检测        |
| `StageDivider.detect_fo_stages`            | 前100点初始化 + 全局百分位阈值                | 依赖初始段 + 全局排序  | 滑动窗口统计量 + 自适应基线 |
| `StageDivider.fuse_stages`                 | 全局排序跳变幅度分配阶段                      | 需要所有跳变点才能排序 | 在线状态机：逐点决策        |
| `AnomalyDetector.detect_strain_anomaly`    | `np.percentile(sd, 99.5)`                   | 全局百分位             | 滑动窗口百分位              |
| `AnomalyDetector.detect_ae_anomaly`        | `np.mean(sc) + 3*np.std(sc)`                | 全局统计量             | EWMA 在线均值和方差         |
| `AnomalyDetector.detect_isolation_forest`  | 批处理训练                                    | 需要全部数据           | 滑动窗口 Z-score 替代       |
| `FeatureExtractor.extract_ae_features`     | `cumsum().max()` 全局归一化                 | 需要全局最大值         | 滑动窗口归一化              |

## 4. 在线流式架构设计

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    在线流式处理引擎                            │
│                                                             │
│  数据源 ──► 数据对齐 ──► 滑动窗口 ──► 特征提取 ──► 阶段判定    │
│  (模拟流)     (已对齐)   特征缓冲区     在线特征      状态机    │
│                                          │                  │
│                                          ▼                  │
│                                    异常检测 ──► 结果输出      │
│                                    (在线Z-score)   │         │
│                                                   ▼         │
│                                             实时可视化       │
│                                             (Flask+SSE)     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 核心组件设计

#### 组件1：`OnlineBuffer` — 滑动窗口缓冲区

```python
class OnlineBuffer:
    """维护多个滑动窗口，支持增量更新和在线统计"""
  
    def __init__(self, window_sizes=[50, 100, 200, 500]):
        self.buffers = {w: deque(maxlen=w) for w in window_sizes}
        self.stats = {w: {} for w in window_sizes}
  
    def push(self, value):
        for w, buf in self.buffers.items():
            buf.append(value)
  
    def get_percentile(self, window, q):
        arr = np.array(self.buffers[window])
        if len(arr) < window // 2:
            return None
        return np.percentile(arr, q)
  
    def get_mean_std(self, window):
        arr = np.array(self.buffers[window])
        if len(arr) < 2:
            return None, None
        return np.mean(arr), np.std(arr)
```

#### 组件2：`OnlineNormalizer` — 在线滑动窗口归一化（改造1）

**改造说明**：取消原 `DataLoader.normalize()` 的全局 min-max 归一化，改为在线滑动窗口归一化。每个新数据点到达时，仅基于历史窗口（w=200）的局部 min/max 进行归一化，不窥探未来数据。

```python
class OnlineNormalizer:
    """在线滑动窗口归一化：仅依赖历史数据，不窥探未来"""
  
    def __init__(self, warmup=100):
        self.warmup = warmup          # 预热期：前100点不归一化
        self.count = 0
        self.strain_buffer = OnlineBuffer([200, 500])
        self.ae_buffers = {}          # 每个AE通道独立维护
        self.fo_buffers = {}          # 每个FO通道独立维护
  
    def normalize_strain(self, strain_val):
        # 预热期内返回原始值
        if self.count < self.warmup:
            return strain_val
        # 基于窗口200的局部 min-max 归一化
        cmin, cmax = self.strain_buffer.get_min_max(200)
        if cmin is None or cmax is None or cmax <= cmin:
            return strain_val
        return (strain_val - cmin) / (cmax - cmin)
```

**关键设计**：

- **预热期**（warmup=100）：前100点不归一化，待窗口积累足够数据
- **局部 min-max**：仅基于最近200个点的最小/最大值，而非全局
- **逐通道独立**：每个AE/FO通道有独立的滑动窗口缓冲区

#### 组件3：`OnlineStageDivider` — 在线阶段划分状态机

**核心思想**：阶段是单调递增的（0→1→2→3），不可逆。每个新数据点到达时，判断是否触发阶段跃迁。

```
状态机模型：

  ┌──────────┐    strain跳变/AE能量激增    ┌──────────┐
  │ Phase 0  │ ──────────────────────────► │ Phase 1  │
  │  健康期   │     FO显著偏移              │ 微损伤期  │
  └──────────┘                             └──────────┘
       ▲                                        │
       │                                        │ strain大幅跳变
       │                                        ▼
       │                                   ┌──────────┐
       │  FO/应变综合判断                   │ Phase 2  │
       │                                   │ 扩展期    │
       │                                   └──────────┘
       │                                        │
       │                                        │ 最大应变跳变
       │                                        ▼
       │                                   ┌──────────┐
       │                                   │ Phase 3  │
       │                                   │ 失效期    │
       │                                   └──────────┘
```

**关键设计**：

1. **应变跳变检测（在线版）**：

   - 维护滑动窗口（w=200）的跳变幅度百分位
   - 当新点的 `|diff|` 超过窗口 99% 百分位 × 自适应系数，标记为跳变
   - 跳变幅度超过窗口应变范围 10% 才触发阶段跃迁
2. **AE 累积能量斜率检测**：

   - 在线维护 `ae_cumulative_energy`（累加器）
   - 计算滑动窗口（w=100）内的能量斜率（差分均值）
   - 斜率持续上升超过阈值 → 触发阶段跃迁
3. **FO 基线漂移检测**：

   - 使用 EWMA 估计在线基线：`baseline = α * x + (1-α) * baseline`
   - 当前值偏离基线超过 3×EWMA标准差 → 标记异常
4. **自适应阈值（改造3）**：

   - 系统启动后先收集 **200点基线数据**
   - 基于基线统计量动态计算所有阈值：
     - **应变跳变阈值**：基于变异系数 × 3（下限0.15）
     - **AE斜率比阈值**：基于 (均值 + 3σ) / 均值（下限1.5）
     - **FO漂移阈值**：基于 FO 变异系数 × 4（下限0.15）
     - **冷却周期**：100~300点，基于应变变异系数动态调整
     - **跃迁阈值**：基于基线噪声水平，各阶段独立计算
   - 不同组的数据特性差异大，自适应阈值确保无需手动调参
5. **融合决策**：

   - 不依赖全局排序，而是**逐点加权投票**
   - 各传感器独立输出"阶段跃迁置信度"
   - 加权融合后决定是否跃迁

#### 组件4：`OnlineAnomalyDetector` — 在线异常检测

```python
class OnlineAnomalyDetector:
    def __init__(self):
        self.strain_buffer = OnlineBuffer([50, 200])
        self.ae_score_buffer = OnlineBuffer([100, 500])
        self.fo_buffer = OnlineBuffer([50, 200])
   
    def update(self, strain_val, ae_score, fo_features):
        # 应变：滑动窗口 Z-score > 3.0
        strain_anom = self._zscore_test(strain_val, 'strain', w=50)
        # AE：滑动窗口均值+3σ
        ae_anom = self._ewma_test(ae_score, 'ae')
        # FO：滑动窗口百分位
        fo_anom = self._percentile_test(fo_features, 'fo')
        # 阶段感知融合
        return self._fuse(strain_anom, ae_anom, fo_anom)
```

#### 组件5：`ChunkedDataReader` — 逐块流式读取（改造2）

**改造说明**：原 `StreamSimulator` 在 `load_data()` 中一次性加载全部数据到内存。改为 `ChunkedDataReader` 逐块读取，每块1000行，内存占用降低约 1000×。

```python
class ChunkedDataReader:
    """逐块数据读取器：不预加载全部数据，按需分块读取"""
  
    def __init__(self, group_id, chunk_size=1000):
        self.group_id = group_id
        self.chunk_size = chunk_size
        self._temp_file = None
        self._reader = None
        self._current_chunk = None
        self._chunk_index = 0
        self._total_rows = 0
  
    def load_and_prepare(self):
        """加载原始数据 → 对齐 → 写入临时CSV → 打开流式读取器"""
        dl = DataLoader(self.group_id).load_all()
        data = dl.sync_timeline()
        # 写入临时文件（不保留在内存中）
        self._temp_file = os.path.join(BASE_DIR, f'.temp_{self.group_id}.csv')
        data.to_csv(self._temp_file, index=False)
        self._reader = pd.read_csv(self._temp_file, chunksize=self.chunk_size)
        self._total_rows = len(data)
        del data  # 释放内存
        return self._total_rows
  
    def read_row(self):
        """读取下一行数据，当前块读完时自动加载下一块"""
        if self._current_chunk is None or self._chunk_index >= len(self._current_chunk):
            if not self._load_next_chunk():
                return None
        row = self._current_chunk.iloc[self._chunk_index]
        self._chunk_index += 1
        return row
  
    def cleanup(self):
        """清理临时文件"""
        if self._temp_file and os.path.exists(self._temp_file):
            os.remove(self._temp_file)
```

**关键设计**：

- **`pandas.read_csv(chunksize=1000)`**：惰性迭代器，每次只加载1000行到内存
- **临时文件**：对齐后的数据写入 `.temp_{group_id}.csv`，处理完后自动清理
- **内存安全**：`del data` 显式释放原始 DataFrame

#### 组件6：`StreamSimulator` — 数据流模拟器（改造2重写）

```python
class StreamSimulator:
    def __init__(self, group_id, speed_factor=1.0):
        self.group_id = group_id
        self.speed_factor = speed_factor
        self.reader = ChunkedDataReader(group_id)
        self.index = 0
  
    def load_data(self):
        return self.reader.load_and_prepare()
  
    def next_point(self):
        row = self.reader.read_row()
        if row is None:
            return None
        self.index += 1
        return row
  
    def has_next(self):
        return self.reader._reader is not None
  
    def cleanup(self):
        self.reader.cleanup()
```

### 4.3 在线特征提取

| 离线特征                         | 在线替代方案                   |
| -------------------------------- | ------------------------------ |
| `strain_ma` rolling(50).mean() | 滑动窗口均值（OnlineBuffer）   |
| `strain_std` rolling(50).std() | 滑动窗口标准差（OnlineBuffer） |
| `strain_diff`                  | 直接计算`current - previous` |
| `strain_cumdiff`               | 累加器`cumsum += diff`       |
| `ae_cumulative_energy`         | 累加器`cumsum += peak²`     |
| `ae_cumulative_energy_norm`    | 滑动窗口归一化（窗口最大值）   |
| `ae_event_rate`                | 滑动窗口计数                   |
| `fo_*_zscore`                  | 滑动窗口 Z-score               |

### 4.4 实时可视化系统

使用 **Flask + Server-Sent Events (SSE) + Chart.js** 实现实时仪表盘：

```
┌──────────────────────────────────────────────────────┐
│              实时阶段划分监控系统                        │
├──────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ 当前阶段  │  │ 数据进度  │  │ 异常率   │  │ 处理耗时 │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
├──────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐  │
│  │  应变实时曲线 (Chart.js 滚动更新)                 │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────┐  ┌─────────────────────────┐ │
│  │  阶段分布柱状图       │  │  运行日志面板            │ │
│  └─────────────────────┘  └─────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**技术选型**：

- **后端**：Flask + SSE 推送
- **前端**：Chart.js + 原生 JS
- **数据流**：模拟器逐点推送 → 处理引擎 → SSE → 前端实时更新

### 4.5 处理流程

```
初始化:
  for each group in [指定组列表]:
    1. ChunkedDataReader.load_and_prepare()  # 逐块读取，不预加载
    2. 初始化 OnlineNormalizer               # 滑动窗口归一化
    3. 初始化 OnlineStageDivider             # 含自适应阈值
    4. 初始化 OnlineAnomalyDetector
    5. 初始化 StreamSimulator

流处理循环:
  while simulator.has_next():
    1. point = simulator.next_point()
    2. norm_point = online_normalizer.normalize_all(point)  # 在线归一化
    3. features = extract_features_online(norm_point, buffers)
    4. stage = stage_divider.update(norm_point, features)   # 自适应阈值
    5. anomaly = anomaly_detector.update(norm_point, features)
    6. 推送结果到前端 (SSE)
    7. sleep(1/speed_factor * 采样间隔)

清理:
  simulator.cleanup()  # 删除临时文件
```

## 5. 关键算法设计

### 5.1 在线应变跳变检测

```python
def _detect_strain_jump_online(self, current_val, prev_val):
    diff = abs(current_val - prev_val)
    self.strain_diff_buffer.push(diff)
    if len(self.strain_diff_buffer.buffers[200]) < 100:
        return 0.0
    p99 = self.strain_diff_buffer.get_percentile(200, 99)
    if p99 is None or p99 < 1e-10:
        return 0.0
    jump_ratio = diff / p99
    strain_range = self.strain_buffer.get_percentile(200, 99) - \
                   self.strain_buffer.get_percentile(200, 1)
    if strain_range > 0 and diff > strain_range * 0.1:
        return jump_ratio
    return 0.0
```

### 5.2 在线 AE 阶段检测

```python
def _detect_ae_slope_change(self, ae_peak):
    self.energy_accumulator += ae_peak ** 2
    if len(self.energy_slope_buffer) > 0:
        prev = self.energy_slope_buffer[-1]
        slope = self.energy_accumulator - prev
    else:
        slope = 0
    self.energy_slope_buffer.append(self.energy_accumulator)
    if len(self.energy_slope_buffer) >= 50:
        recent_slopes = list(self.energy_slope_buffer)[-50:]
        slope_changes = np.diff(recent_slopes)
        mean_slope_change = np.mean(slope_changes)
        if mean_slope_change > 0 and slope > np.percentile(recent_slopes, 90):
            return True
    return False
```

### 5.3 在线 FO 漂移检测

```python
def _detect_fo_drift(self, fo_values):
    fo_mean = np.mean(fo_values)
    if self.ewma_baseline is None:
        self.ewma_baseline = fo_mean
        self.ewma_std = 0.0
        return 0.0
    residual = fo_mean - self.ewma_baseline
    self.ewma_baseline = self.alpha * fo_mean + (1 - self.alpha) * self.ewma_baseline
    self.ewma_std = np.sqrt((1 - self.alpha) * (self.ewma_std ** 2 + 
                            self.alpha * residual ** 2))
    if self.ewma_std > 1e-10:
        return abs(residual) / self.ewma_std
    return 0.0
```

### 5.4 自适应阈值计算（改造3核心）

```python
def _compute_adaptive_thresholds(self):
    """基于基线统计量动态计算所有阈值"""
    # 1. 应变跳变阈值: 基于变异系数 × 3，下限0.15
    strain_arr = np.array(self.strain_baseline_values)
    strain_std = float(np.std(strain_arr))
    strain_mean = float(np.mean(strain_arr))
    if strain_mean > 1e-6:
        cv = strain_std / strain_mean
        self.strain_jump_threshold = max(3.0 * cv, 0.15)
  
    # 2. AE斜率比阈值: 基于 (均值 + 3σ) / 均值，下限1.5
    if len(self.ae_slope_baseline) > 20:
        slope_arr = np.array(self.ae_slope_baseline)
        slope_mean = float(np.mean(slope_arr))
        slope_std = float(np.std(slope_arr))
        self.ae_slope_ratio_threshold = max(
            (slope_mean + 3.0 * slope_std) / slope_mean, 1.5)
  
    # 3. FO漂移阈值: 基于 FO 变异系数 × 4，下限0.15
    if fo_mean > 1e-6:
        fo_cv = fo_std / fo_mean
        self.fo_drift_threshold = max(4.0 * fo_cv, 0.15)
  
    # 4. 冷却周期: 基于应变变异系数动态调整 (100~300)
    self.cooldown_period = int(max(100, min(300, 100 + base_cv * 1000)))
  
    # 5. 跃迁阈值: 基于基线噪声水平动态调整
    self.transition_thresholds = {
        0: max(0.15, min(0.50, 0.25 * noise_factor)),
        1: max(0.20, min(0.60, 0.30 * noise_factor)),
        2: max(0.30, min(0.70, 0.40 * noise_factor)),
    }
```

### 5.5 在线阶段跃迁决策

```python
def _decide_transition(self, strain_jump, ae_slope, fo_drift):
    confidence = {
        'strain': min(strain_jump / 3.0, 1.0),
        'ae': 0.8 if ae_slope else 0.0,
        'fo': min(fo_drift / 3.0, 1.0),
    }
    weights = {
        0: {'strain': 0.3, 'ae': 0.5, 'fo': 0.2},
        1: {'strain': 0.4, 'ae': 0.4, 'fo': 0.2},
        2: {'strain': 0.5, 'ae': 0.3, 'fo': 0.2},
        3: {'strain': 0.6, 'ae': 0.1, 'fo': 0.3},
    }
    w = weights.get(self.current_phase, weights[1])
    fusion_score = sum(confidence[k] * w[k] for k in ['strain', 'ae', 'fo'])
    # 使用自适应阈值替代固定阈值
    threshold = self.transition_thresholds.get(self.current_phase, 0.5)
    if fusion_score > threshold:
        self.current_phase += 1
```

## 6. 文件结构

> **【拆分】已将原 `multi_source_shm.py`（1455 行）拆分为模块化的 `shm` 包**，前端 HTML 也拆到独立模板文件，可读性大幅提升。

```
multi_source_shm.py  (兼容入口，仅 re-export shm 包，保留旧导入方式)

shm/                          # 主包
├── __init__.py               # 导出主接口 + 入口函数
├── config.py                 # 常量、路径、全局配置（集中管理所有阈值/窗口参数）
├── data_loader.py            # DataLoader — 数据加载与时间同步
├── online_buffer.py          # OnlineBuffer — 滑动窗口缓冲区
├── normalizer.py             # OnlineNormalizer — 在线滑动窗口归一化（改造1）
├── feature_extractor.py      # OnlineFeatureExtractor — 逐点特征提取
├── stage_divider.py          # OnlineStageDivider — 阶段划分状态机（自适应阈值+注意力融合）
├── anomaly_detector.py       # OnlineAnomalyDetector — 在线异常检测
├── streaming.py              # ChunkedDataReader + StreamSimulator + StreamProcessor
├── dashboard.py              # RealtimeDashboard — Flask SSE 实时仪表盘
└── templates/
    └── dashboard.html        # 前端 HTML 模板（Chart.js，从原 HTML_TEMPLATE 拆出）
```

**各模块职责：**

| 模块                         | 类/函数                    | 职责                                             |
| ---------------------------- | -------------------------- | ------------------------------------------------ |
| `config.py`                | 常量                       | 路径、默认组号、阈值、窗口、分块大小、仪表盘参数 |
| `data_loader.py`           | `DataLoader`             | 加载应变、AE、光纤数据，时间同步                 |
| `online_buffer.py`         | `OnlineBuffer`           | 滑动窗口缓冲区，支持多窗口大小                   |
| `normalizer.py`            | `OnlineNormalizer`       | 在线滑动窗口归一化（改造1：替代全局归一化）      |
| `feature_extractor.py`     | `OnlineFeatureExtractor` | 在线特征提取                                     |
| `stage_divider.py`         | `OnlineStageDivider`     | 在线阶段划分状态机（含自适应阈值，改造3）        |
| `anomaly_detector.py`      | `OnlineAnomalyDetector`  | 在线异常检测                                     |
| `streaming.py`             | `ChunkedDataReader`      | 逐块数据读取器（改造2：替代全量加载）            |
| `streaming.py`             | `StreamSimulator`        | 数据流模拟器（基于ChunkedDataReader）            |
| `streaming.py`             | `StreamProcessor`        | 在线流式处理主引擎                               |
| `dashboard.py`             | `RealtimeDashboard`      | 实时可视化仪表盘（Flask + SSE + Chart.js）       |
| `templates/dashboard.html` | 前端模板                   | Chart.js 图表 + SSE 事件处理                     |

**入口函数（位于 `shm/__init__.py`）：**

```python
run_online_dashboard()    # 启动实时仪表盘
run_online_processing()   # 批量在线处理（无界面）
main()                    # 命令行入口
```

## 7. 实施步骤

| 步骤         | 内容                                                                                                             | 状态                |
| ------------ | ---------------------------------------------------------------------------------------------------------------- | ------------------- |
| 1            | 实现`OnlineBuffer` 滑动窗口缓冲区                                                                              | ✅ 已完成           |
| 2            | 实现`OnlineFeatureExtractor` 在线特征提取                                                                      | ✅ 已完成           |
| 3            | 实现`OnlineStageDivider` 在线阶段划分状态机                                                                    | ✅ 已完成           |
| 4            | 实现`OnlineAnomalyDetector` 在线异常检测                                                                       | ✅ 已完成           |
| 5            | 实现`StreamSimulator` 数据流模拟器                                                                             | ✅ 已完成           |
| 6            | 实现`StreamProcessor` 流式处理主引擎                                                                           | ✅ 已完成           |
| 7            | 实现`RealtimeDashboard` 实时可视化仪表盘                                                                       | ✅ 已完成           |
| 8            | 集成测试：对 016-020 运行在线流程                                                                                | ✅ 已完成           |
| 9            | 清理离线代码和文档                                                                                               | ✅ 已完成           |
| **10** | **改造1：取消全局归一化 → `OnlineNormalizer` 滑动窗口归一化**                                           | **✅ 已完成** |
| **11** | **改造2：`ChunkedDataReader` 逐块流式读取替代全量加载**                                                  | **✅ 已完成** |
| **12** | **改造3：自适应阈值（基线统计量动态计算）**                                                                | **✅ 已完成** |
| **13** | **修复4：三源真正融合 — 应变虚高修复、FO Z-score保留漂移信号、AE多指标融合、注意力权重**                  | **✅ 已完成** |
| **14** | **修复5：阶段跃迁过早 — 最小稳定期500点、基线500点、冷却300~800点、阈值大幅提高**                         | **✅ 已完成** |
| **15** | **修复6：Phase 2→3不触发 — AE event_rate上限从0.5降到0.3、spike检测门槛从0.05降到0.03、Phase 2阈值降低** | **✅ 已完成** |
| **16** | **拆分：将 multi_source_shm.py（1455行）拆分为 shm 包，前端 HTML 拆到 templates/dashboard.html**           | **✅ 已完成** |
| **17** | **拆分：创建 shm/__init__.py 导出主接口，保留 multi_source_shm.py 作为兼容入口**                     | **✅ 已完成** |
| **18** | **拆分：更新 test_online.py 导入，验证拆分后系统正常运行**                                                 | **✅ 已完成** |

## 8. 运行方式

### 8.1 实时仪表盘模式

```bash
# 运行单组（默认 016）
python multi_source_shm.py dashboard --group 016 --speed 10 --port 5000

# 运行其他组（如 019 关键故障组）
python multi_source_shm.py dashboard --group 019 --speed 10 --port 5001

# 运行任意组（001-027 均可）
python multi_source_shm.py dashboard --group 021 --speed 10 --port 5002
```

参数说明：

- `--group`：组号（默认 016，支持 001-027 任意组号）
- `--speed`：模拟速度倍率（默认 10）
- `--port`：仪表盘端口（默认 5000）

### 8.2 批量处理模式（无界面）

```bash
# 处理默认五组
python multi_source_shm.py batch --groups 016 017 018 019 020 --speed 10

# 处理自定义组列表
python multi_source_shm.py batch --groups 021 022 023 --speed 10

# 处理单组
python multi_source_shm.py batch --groups 016 --speed 10
```

### 8.3 测试脚本

```bash
# 测试默认组 016
python test_online.py

# 测试指定组（如 019）
python test_online.py --group 019

# 测试指定组并限制处理点数
python test_online.py --group 021 --max-points 2000
```

### 8.4 组号支持说明

> **所有 `'016'` 引用均为默认参数值，非硬编码限制。**
>
> 系统通过 `DataLoader` 动态构造文件路径，支持任意组号：
>
> - 数据目录结构：`{group_id}/{group_id}应变.csv`、`{group_id}/{group_id}声发射.csv`、`{group_id}/{group_id}光纤.csv`
> - 支持 `.csv` 和 `.xlsx` 格式自动检测
> - 缺失的传感器通道自动跳过，不影响处理流程
>
> 已验证可运行的组：016, 017, 018, 019, 020（有完整三传感器数据）
> 理论上支持：001-027（取决于各组是否有对应的数据文件）

### 8.5 代码导入方式（拆分后）

> **【拆分】`multi_source_shm.py` 已拆分为 `shm` 包**，推荐新代码直接使用 `shm` 包：

```python
# 推荐：直接使用 shm 包
from shm import StreamProcessor, RealtimeDashboard
from shm import run_online_dashboard, run_online_processing

# 兼容：旧导入方式依然可用（multi_source_shm.py 会 re-export shm 包）
from multi_source_shm import StreamProcessor
```

命令行运行方式不变（`multi_source_shm.py` 仍为入口）：

```bash
# 实时仪表盘
python multi_source_shm.py dashboard --group 016 --speed 10 --port 5000

# 批量处理
python multi_source_shm.py batch --groups 016 017 018 019 020 --speed 10

# 快速测试
python test_online.py --group 016 --max-points 600
```

## 9. 算法参数说明

### 9.1 阶段划分参数（自适应）

| 参数            | 值                                                   | 说明                                                               |
| --------------- | ---------------------------------------------------- | ------------------------------------------------------------------ |
| 应变滑动窗口    | 50, 100, 200, 500                                    | 多尺度窗口                                                         |
| AE能量窗口      | 50, 100, 200, 500                                    | 能量累积和斜率检测                                                 |
| FO均值窗口      | 50, 100, 200, 500                                    | 基线漂移检测                                                       |
| 基线采集期      | 500 点                                               | 用于计算自适应阈值（修复5：从200增加到500）                        |
| 最小稳定期      | 500 点                                               | 前500点不允许任何跃迁（修复5新增）                                 |
| 最小阶段持续    | 300 点                                               | 每个阶段至少维持300点才能再次跃迁（修复5新增）                     |
| 跃迁冷却期      | **自适应 300~800 点**                          | 基于应变变异系数动态调整（修复5：从100~300增加）                   |
| Phase 0→1 阈值 | **自适应 0.40~0.70**                           | 基于基线噪声水平（修复5：从0.15~0.50提高）                         |
| Phase 1→2 阈值 | **自适应 0.45~0.75**                           | 基于基线噪声水平（修复5：从0.20~0.60提高）                         |
| Phase 2→3 阈值 | **自适应 0.40~0.70**                           | 基于基线噪声水平（修复6：从0.50~0.80降低，允许AE尖峰触发）         |
| 应变跳变阈值    | **自适应 CV×5（下限0.3，上限1.5）**           | 基于应变变异系数（修复5：从CV×3/0.15提高）                        |
| AE斜率比阈值    | **自适应 (均值+5σ)/均值（下限2.0，上限8.0）** | 基于AE基线斜率（修复5：从3σ/1.5提高）                             |
| FO漂移阈值      | **自适应 FO-CV×6（下限0.3，上限1.0）**        | 基于FO变异系数（修复5：从CV×4/0.15提高；修复6：上限从1.5降到1.0） |

### 9.2 异常检测参数

| 参数                 | 值       | 说明     |
| -------------------- | -------- | -------- |
| 应变 Z-score 阈值    | 3.0      | 滑动窗口 |
| 应变跳变百分位       | 99.5%    | 滑动窗口 |
| AE 异常分数阈值      | 均值+3σ | 滑动窗口 |
| AE 峰度百分位        | 99%      | 滑动窗口 |
| FO 范围百分位        | 99%      | 滑动窗口 |
| FO 通道 Z-score 阈值 | 3.0      | 滑动窗口 |

### 9.3 在线归一化参数（改造1）

| 参数          | 值     | 说明                        |
| ------------- | ------ | --------------------------- |
| 预热期 warmup | 100 点 | 前100点不归一化，返回原始值 |
| 归一化窗口    | 200    | 滑动 min-max 归一化窗口大小 |
| 归一化范围    | [0, 1] | 局部 min-max 映射           |

### 9.4 流式读取参数（改造2）

| 参数              | 值                       | 说明                           |
| ----------------- | ------------------------ | ------------------------------ |
| 块大小 chunk_size | 1000 行                  | 每批加载行数                   |
| 临时文件          | `.temp_{group_id}.csv` | 对齐后数据缓存，处理完自动删除 |
