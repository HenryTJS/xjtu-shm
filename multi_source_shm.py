#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞行器结构健康监测 — 在线流式处理系统
融合应变、声发射(AE)、光纤(FO)三种传感器数据，实现：
  1. 在线故障阶段划分（健康期 → 微损伤萌生 → 损伤扩展 → 结构失效）
  2. 在线异常点识别（多传感器融合决策）
  3. 实时可视化仪表盘（Flask + SSE + Chart.js）

仅处理 016-020 五组已对齐数据，逐点流式处理。

【改造说明】
  1. 取消全局归一化 → 滑动窗口在线归一化（OnlineNormalizer）
  2. StreamSimulator 改为逐块流式读取（chunked streaming）
  3. 自适应阈值（基于基线统计量动态计算）
"""

import os, warnings, json, queue, threading
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 数据加载器（仅负责加载和对齐，不再做全局归一化）
# ============================================================
class DataLoader:
    """数据加载器：统一加载应变、声发射、光纤数据，并进行时间同步"""

    def __init__(self, group_id):
        self.group_id = str(group_id).zfill(3)
        self.data_dir = os.path.join(BASE_DIR, self.group_id)
        self.strain = self.ae = self.fo = self.synced_data = None

    def _detect_encoding(self, file_path):
        for enc in ['utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    f.readline()
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        return 'latin-1'

    def _find_file(self, name_patterns):
        for p in name_patterns:
            fp = os.path.join(self.data_dir, p)
            if os.path.exists(fp):
                return fp
        return None

    def _unify_columns(self, df, col_map_rules):
        df.columns = df.columns.str.strip()
        col_map = {}
        for col in df.columns:
            cl = col.lower().replace(' ', '').replace('(', '').replace(')', '').replace('（', '').replace('）', '').replace('_', '')
            for pattern, target in col_map_rules:
                if pattern in cl:
                    col_map[col] = target
                    break
        for i, (pattern, target) in enumerate([('time', 'time'), ('strain', 'strain')]):
            if target not in col_map.values() and len(df.columns) > i:
                col_map[df.columns[i]] = target
        return df.rename(columns=col_map)

    def load_strain(self):
        fp = self._find_file([f'{self.group_id}应变.csv', f'{self.group_id}应变.xlsx'])
        if fp is None:
            print(f'  [警告] 未找到应变文件: {self.data_dir}')
            return None
        if fp.endswith('.csv'):
            encoding = self._detect_encoding(fp)
            with open(fp, 'r', encoding=encoding) as f:
                sep = '\t' if '\t' in f.readline() else ','
            df = pd.read_csv(fp, sep=sep, encoding=encoding, engine='python')
        else:
            df = pd.read_excel(fp)
        df = self._unify_columns(df, [
            (c, 'time') for c in ['时间', 'time', 'timestamp']
        ] + [
            (c, 'strain') for c in ['应变', 'strain', '幅值']
        ])
        df = df[['time', 'strain']].copy()
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['strain'] = pd.to_numeric(df['strain'], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        self.strain = df
        return df

    def load_ae(self):
        fp = os.path.join(self.data_dir, f'{self.group_id}声发射.csv')
        if not os.path.exists(fp):
            print(f'  [警告] 未找到声发射文件: {fp}')
            return None
        df = pd.read_csv(fp, encoding='utf-8-sig', engine='python')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        self.ae = df
        return df

    def load_fo(self):
        fp = self._find_file([f'{self.group_id}光纤.csv', f'{self.group_id}光纤.xlsx'])
        if fp is None:
            print(f'  [警告] 未找到光纤文件: {self.data_dir}')
            return None
        df = pd.read_csv(fp, encoding='utf-8-sig', engine='python') if fp.endswith('.csv') else pd.read_excel(fp)
        df.columns = df.columns.str.strip()
        col_map = {}
        for col in df.columns:
            cl = col.lower().replace(' ', '').replace('(', '').replace(')', '').replace('（', '').replace('）', '').replace('_', '')
            if any(t in cl for t in ['时间', 'time', 'timestamp']):
                col_map[col] = 'time'
            elif cl.startswith('s') and len(col.strip()) <= 3:
                col_map[col] = col.strip()
            elif 'fiber' in cl and 's' in cl:
                col_map[col] = col.split('_')[-1]
        if 'time' not in col_map.values() and len(df.columns) > 0:
            col_map[df.columns[0]] = 'time'
        df = df.rename(columns=col_map)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        self.fo = df
        return df

    def load_all(self):
        print(f'\n>>> 加载组 {self.group_id} 数据')
        self.load_strain(); self.load_ae(); self.load_fo()
        return self

    def sync_timeline(self):
        if self.strain is None:
            return None
        timeline = self.strain[['time']].copy().sort_values('time').drop_duplicates('time').reset_index(drop=True)
        timeline['time'] = timeline['time'].astype(float)
        result = timeline.merge(self.strain, on='time', how='left')
        if self.fo is not None:
            fo_t = self.fo[['time']].copy().sort_values('time').drop_duplicates('time').reset_index(drop=True)
            fo_t['time'] = fo_t['time'].astype(float)
            td = np.diff(np.sort(result['time'].unique()))
            tol = np.median(td) * 1.5 if len(td) > 0 else 1.0
            result = pd.merge_asof(result.sort_values('time'), self.fo.sort_values('time').astype({'time': float}),
                                   on='time', direction='nearest', tolerance=tol)
            fo_cols = [c for c in result.columns if c.startswith('s') and len(c) <= 3]
            if fo_cols:
                result[fo_cols] = result[fo_cols].ffill()
        if self.ae is not None:
            ae_time = np.linspace(result['time'].min(), result['time'].max(), len(self.ae))
            target = result['time'].values
            ae_vals = self.ae.values.T
            interp = np.array([np.interp(target, ae_time, col) for col in ae_vals]).T
            for i, col in enumerate(self.ae.columns):
                result[f'ae_{col}'] = interp[:, i]
        self.synced_data = result
        return result

    # ========== 改造1: 移除全局归一化方法 ==========
    # normalize() 已删除，改为 OnlineNormalizer 在线滑动窗口归一化


# ============================================================
# 在线流式处理模块
# ============================================================

class OnlineBuffer:
    """滑动窗口缓冲区，支持多窗口大小，在线计算统计量"""

    def __init__(self, window_sizes=None):
        if window_sizes is None:
            window_sizes = [50, 100, 200, 500]
        self.buffers = {w: [] for w in window_sizes}
        self.maxlens = {w: w for w in window_sizes}

    def push(self, value):
        for w in self.buffers:
            self.buffers[w].append(value)
            if len(self.buffers[w]) > self.maxlens[w]:
                self.buffers[w].pop(0)

    def __len__(self, w=None):
        if w is not None:
            return len(self.buffers.get(w, []))
        return max(len(v) for v in self.buffers.values()) if self.buffers else 0

    def is_warm(self, w, min_ratio=0.5):
        return len(self.buffers.get(w, [])) >= w * min_ratio

    def get_array(self, w):
        return np.array(self.buffers.get(w, []))

    def get_percentile(self, w, q):
        arr = self.get_array(w)
        if len(arr) < max(5, w * 0.3):
            return None
        return float(np.percentile(arr, q))

    def get_mean_std(self, w):
        arr = self.get_array(w)
        if len(arr) < 2:
            return None, None
        return float(np.mean(arr)), float(np.std(arr))

    def get_min_max(self, w):
        arr = self.get_array(w)
        if len(arr) == 0:
            return None, None
        return float(np.min(arr)), float(np.max(arr))

    def get_max(self, w=None):
        if w is not None:
            arr = self.get_array(w)
            return float(np.max(arr)) if len(arr) > 0 else 0.0
        w = next(iter(self.buffers.keys()))
        arr = self.get_array(w)
        return float(np.max(arr)) if len(arr) > 0 else 0.0


# ========== 改造1: 新增 OnlineNormalizer 类 ==========
class OnlineNormalizer:
    """在线滑动窗口归一化：仅依赖历史数据，不窥探未来

    对每种传感器类型维护独立的滑动窗口缓冲区：
    - 应变: 使用 OnlineBuffer 滚动 min/max 进行 min-max 归一化
    - AE:   先 log1p 变换，再滚动 min-max 归一化
    - FO:   使用 Z-score 标准化（保留漂移信息），而非 min-max 压缩

    在 warm-up 阶段（窗口未满）返回原始值，避免早期不稳定。
    """

    def __init__(self, warmup=100):
        self.warmup = warmup
        self.count = 0
        # 应变
        self.strain_buffer = OnlineBuffer([200, 500])
        # AE 各通道
        self.ae_buffers = {}
        # FO 各通道 — 使用 Z-score 标准化，保留漂移信息
        self.fo_buffers = {}

    def _get_ae_buffer(self, name):
        if name not in self.ae_buffers:
            self.ae_buffers[name] = OnlineBuffer([200, 500])
        return self.ae_buffers[name]

    def _get_fo_buffer(self, name):
        if name not in self.fo_buffers:
            self.fo_buffers[name] = OnlineBuffer([200, 500])
        return self.fo_buffers[name]

    def normalize_strain(self, strain_val):
        """在线归一化应变值，返回 (归一化值, 原始值)"""
        if strain_val is None or np.isnan(strain_val):
            return strain_val, strain_val
        self.strain_buffer.push(strain_val)
        self.count += 1
        raw = strain_val
        if self.count < self.warmup or not self.strain_buffer.is_warm(200):
            return raw, raw  # warm-up 阶段返回原始值
        cmin, cmax = self.strain_buffer.get_min_max(200)
        if cmin is None or cmax is None or cmax <= cmin:
            return raw, raw
        norm = (raw - cmin) / (cmax - cmin)
        return norm, raw  # 返回 (归一化值, 原始值)

    def normalize_ae(self, ae_dict):
        """在线归一化 AE 各通道"""
        if not ae_dict:
            return ae_dict
        result = {}
        for key, val in ae_dict.items():
            # log1p 变换
            cd = np.log1p(max(val, 0))
            buf = self._get_ae_buffer(key)
            buf.push(cd)
            if self.count < self.warmup or not buf.is_warm(200):
                result[key] = cd  # warm-up 返回 log1p 值
            else:
                cmin, cmax = buf.get_min_max(200)
                if cmin is not None and cmax is not None and cmax > cmin:
                    result[key] = (cd - cmin) / (cmax - cmin)
                else:
                    result[key] = cd
        return result

    def normalize_fo(self, fo_dict):
        """在线 Z-score 标准化 FO 各通道（保留漂移信息，不压缩到 [0,1]）"""
        if not fo_dict:
            return fo_dict
        result = {}
        for key, val in fo_dict.items():
            buf = self._get_fo_buffer(key)
            buf.push(val)
            if self.count < self.warmup or not buf.is_warm(200):
                result[key] = val  # warm-up 返回原始值
            else:
                # Z-score 标准化：保留漂移方向和幅度
                m, s = buf.get_mean_std(200)
                if s is not None and s > 1e-10:
                    result[key] = (val - m) / s  # Z-score，漂移信号保留
                else:
                    result[key] = val
        return result

    def normalize_all(self, strain_val, ae_dict, fo_dict):
        """统一在线归一化入口，返回 (归一化应变, 原始应变, 归一化AE, 归一化FO)"""
        strain_norm, strain_raw = self.normalize_strain(strain_val)
        ae_norm = self.normalize_ae(ae_dict)
        fo_norm = self.normalize_fo(fo_dict)
        return strain_norm, strain_raw, ae_norm, fo_norm


class OnlineFeatureExtractor:
    """在线特征提取：维护滑动窗口状态，逐点计算特征"""

    def __init__(self):
        self.strain_buffer = OnlineBuffer([50, 200])
        self.strain_diff_buffer = OnlineBuffer([50, 200])
        self.strain_cumdiff = 0.0
        self.prev_strain = None
        self.ae_buffers = {}
        self.ae_energy_accum = 0.0
        self.ae_energy_max = 0.0
        self.ae_energy_window = OnlineBuffer([500])
        self.ae_event_count = 0
        self.fo_buffers = {}
        self.fo_mean_buffer = OnlineBuffer([50, 200])
        self.fo_std_buffer = OnlineBuffer([50, 200])
        self.index = 0

    def _get_ae_buffer(self, name):
        if name not in self.ae_buffers:
            self.ae_buffers[name] = OnlineBuffer([50, 100, 200])
        return self.ae_buffers[name]

    def _get_fo_buffer(self, name):
        if name not in self.fo_buffers:
            self.fo_buffers[name] = OnlineBuffer([50, 200])
        return self.fo_buffers[name]

    def extract_strain(self, strain_val):
        feats = {}
        if strain_val is None or np.isnan(strain_val):
            return feats
        self.strain_buffer.push(strain_val)
        feats['strain_raw'] = strain_val
        if self.strain_buffer.is_warm(50):
            m, _ = self.strain_buffer.get_mean_std(50)
            feats['strain_ma'] = m
        else:
            feats['strain_ma'] = strain_val
        if self.strain_buffer.is_warm(50):
            _, s = self.strain_buffer.get_mean_std(50)
            feats['strain_std'] = s if s is not None else 0.0
        else:
            feats['strain_std'] = 0.0
        if self.prev_strain is not None:
            diff = abs(strain_val - self.prev_strain)
        else:
            diff = 0.0
        feats['strain_diff'] = diff
        self.strain_diff_buffer.push(diff)
        self.strain_cumdiff += diff
        feats['strain_cumdiff'] = self.strain_cumdiff
        feats['strain_deviation'] = abs(strain_val - feats['strain_ma'])
        if feats['strain_std'] > 1e-10:
            feats['strain_zscore'] = feats['strain_deviation'] / feats['strain_std']
        else:
            feats['strain_zscore'] = 0.0
        self.prev_strain = strain_val
        return feats

    def extract_ae(self, ae_dict):
        feats = {}
        if not ae_dict:
            return feats
        key_map = {'ae_Peak': 'ae_peak', 'ae_Kurtosis': 'ae_kurtosis',
                   'ae_RMS': 'ae_rms', 'ae_SpectralEnergy': 'ae_spectral_energy',
                   'ae_Mean': 'ae_mean', 'ae_Std': 'ae_std',
                   'ae_Skewness': 'ae_skewness', 'ae_PeakToPeak': 'ae_peak_to_peak'}
        for orig_key, short_key in key_map.items():
            if orig_key in ae_dict:
                val = ae_dict[orig_key]
                feats[short_key] = val
                self._get_ae_buffer(short_key).push(val)
        score = 0.0; ws = 0.0
        for feat, w in [('ae_peak', 0.3), ('ae_kurtosis', 0.3), ('ae_rms', 0.2), ('ae_spectral_energy', 0.2)]:
            if feat in feats:
                score += w * feats[feat]; ws += w
        if ws > 0:
            feats['ae_anomaly_score'] = score / ws
        if 'ae_kurtosis' in feats:
            self.ae_event_count = self.ae_event_count * 0.99 + (1.0 if feats['ae_kurtosis'] > 0.5 else 0.0)
            feats['ae_event_rate'] = self.ae_event_count
        if 'ae_peak' in feats:
            self.ae_energy_accum += feats['ae_peak'] ** 2
            feats['ae_cumulative_energy'] = self.ae_energy_accum
            feats['ae_cumulative_energy_norm'] = self.ae_energy_accum / (self.ae_energy_accum + 1.0)
        for feat_name in ['ae_peak', 'ae_rms', 'ae_mean', 'ae_spectral_energy']:
            if feat_name in feats:
                buf = self._get_ae_buffer(feat_name)
                if buf.is_warm(100):
                    arr = buf.get_array(100)
                    if len(arr) >= 10:
                        feats[f'{feat_name}_kurt'] = float(pd.Series(arr).kurtosis())
                        feats[f'{feat_name}_skew'] = float(pd.Series(arr).skew())
                        ma = float(np.mean(arr))
                        if ma > 1e-10:
                            feats[f'{feat_name}_impulse'] = abs(feats[feat_name]) / ma
        return feats

    def extract_fo(self, fo_dict):
        feats = {}
        if not fo_dict:
            return feats
        fo_values = []
        for ch_name, ch_val in fo_dict.items():
            feats[f'fo_{ch_name}'] = ch_val
            fo_values.append(ch_val)
            self._get_fo_buffer(ch_name).push(ch_val)
        if fo_values:
            fv = np.array(fo_values)
            feats['fo_mean'] = float(np.mean(fv))
            feats['fo_std'] = float(np.std(fv))
            feats['fo_max'] = float(np.max(fv))
            feats['fo_min'] = float(np.min(fv))
            feats['fo_range'] = feats['fo_max'] - feats['fo_min']
            feats['fo_channel_dev'] = float(np.std(fv - feats['fo_mean']))
            self.fo_mean_buffer.push(feats['fo_mean'])
            self.fo_std_buffer.push(feats['fo_std'])
            for ch_name in fo_dict:
                buf = self._get_fo_buffer(ch_name)
                if buf.is_warm(50):
                    m, s = buf.get_mean_std(50)
                    if s is not None and s > 1e-10:
                        feats[f'fo_{ch_name}_zscore'] = (fo_dict[ch_name] - m) / s
        return feats

    def extract_all(self, strain_val, ae_dict, fo_dict):
        feats = {}
        feats.update(self.extract_strain(strain_val))
        feats.update(self.extract_ae(ae_dict))
        feats.update(self.extract_fo(fo_dict))
        self.index += 1
        return feats


class OnlineStageDivider:
    """在线阶段划分状态机：逐点决策，单调跃迁 0→1→2→3
    基于数据驱动策略：
    - 应变: 滑动均值 LEVEL 变化 + 原始值阈值
    - AE:   累积能量加速比（短窗/长窗斜率比）
    - FO:   均值偏移比例

    【改造3】自适应阈值：基于基线（warm-up）统计量动态计算阈值
    """

    def __init__(self):
        self.current_phase = 0
        self.phase_history = []
        # 应变
        self.strain_buffer = OnlineBuffer([50, 100, 200, 500])
        self.strain_baseline_mean = None
        self.strain_warm_count = 0
        # AE
        self.ae_energy_buffer = OnlineBuffer([50, 100, 200, 500])
        self.energy_accum = 0.0
        self.energy_max = 0.0
        self.prev_energy = 0.0
        self.energy_slopes = []
        # FO
        self.fo_mean_buffer = OnlineBuffer([50, 100, 200, 500])
        self.fo_baseline_mean = None
        self.fo_warm_count = 0
        # 跃迁控制
        self.transition_cooldown = 0
        # ========== 改造3: 自适应阈值 ==========
        # 基线统计量（warm-up 阶段收集）
        self.strain_baseline_values = []      # 应变基线值列表
        self.strain_diff_baseline = []        # 应变差分基线
        self.fo_baseline_values = []          # FO 基线值列表
        self.ae_slope_baseline = []           # AE 斜率基线
        self.baseline_collected = False       # 基线是否收集完成
        self.baseline_length = 500            # 基线长度（从200增加到500，确保基线稳定）
        # 自适应阈值（由基线统计量计算得出）
        self.strain_jump_threshold = 0.5      # 默认值（从0.3提高到0.5，防止早期误触发）
        self.ae_slope_ratio_threshold = 3.0   # 默认值（从2.0提高到3.0）
        self.fo_drift_threshold = 0.5         # 默认值（从0.25提高到0.5）
        self.cooldown_period = 500            # 默认值（从200提高到500）
        self.transition_thresholds = {0: 0.60, 1: 0.65, 2: 0.60}  # Phase 2 默认阈值降低到 0.60
        # 基线变化率（用于自适应 cooldown）
        self.baseline_strain_std = None
        self.baseline_fo_std = None
        # ========== 修复5: 强制最小稳定期 ==========
        self.min_stable_points = 500          # 前500点不允许任何跃迁
        self.phase_entry_points = {0: 0}      # 记录每个阶段的进入点
        # 调试
        self.total_points = 0
        self.debug_info = []

    def _update_buffers(self, strain_val, features):
        if strain_val is not None and not np.isnan(strain_val):
            self.strain_buffer.push(strain_val)
        if 'ae_cumulative_energy' in features:
            self.energy_accum = features['ae_cumulative_energy']
            self.ae_energy_buffer.push(self.energy_accum)
            if self.energy_accum > self.energy_max:
                self.energy_max = self.energy_accum
        if 'fo_mean' in features:
            self.fo_mean_buffer.push(features['fo_mean'])

    # ========== 改造3: 自适应基线收集 ==========
    def _collect_baseline(self, strain_val, features):
        """收集 warm-up 阶段的基线数据"""
        if self.baseline_collected:
            return

        # 收集应变基线
        if strain_val is not None and not np.isnan(strain_val):
            self.strain_baseline_values.append(strain_val)
            if len(self.strain_baseline_values) >= 2:
                self.strain_diff_baseline.append(
                    abs(strain_val - self.strain_baseline_values[-2]))

        # 收集 FO 基线
        if 'fo_mean' in features:
            self.fo_baseline_values.append(features['fo_mean'])

        # 收集 AE 斜率基线
        if 'ae_cumulative_energy' in features:
            current_energy = features['ae_cumulative_energy']
            slope = current_energy - self.prev_energy if self.prev_energy > 0 else 0.0
            self.prev_energy = current_energy
            if slope > 0:
                self.ae_slope_baseline.append(slope)

        # 判断基线是否收集完成
        if (len(self.strain_baseline_values) >= self.baseline_length and
                len(self.fo_baseline_values) >= self.baseline_length // 2):
            self._compute_adaptive_thresholds()
            self.baseline_collected = True
            print(f'    [自适应阈值] 基线收集完成, 计算阈值:')
            print(f'      应变跳变阈值={self.strain_jump_threshold:.3f}, '
                  f'AE斜率比阈值={self.ae_slope_ratio_threshold:.2f}, '
                  f'FO漂移阈值={self.fo_drift_threshold:.3f}')
            print(f'      冷却周期={self.cooldown_period}, '
                  f'跃迁阈值={self.transition_thresholds}')

    def _compute_adaptive_thresholds(self):
        """基于基线统计量计算自适应阈值"""
        # --- 应变跳变阈值 ---
        strain_arr = np.array(self.strain_baseline_values)
        strain_std = float(np.std(strain_arr))
        strain_mean = float(np.mean(strain_arr))
        self.baseline_strain_std = strain_std

        # ========== 修复5: 更保守的应变跳变阈值 ==========
        if strain_mean > 1e-6:
            cv = strain_std / strain_mean
            self.strain_jump_threshold = max(5.0 * cv, 0.3)
        else:
            self.strain_jump_threshold = 0.5

        # 应变差分基线
        if len(self.strain_diff_baseline) > 10:
            diff_arr = np.array(self.strain_diff_baseline)
            diff_p95 = float(np.percentile(diff_arr, 95))
            diff_mean = float(np.mean(diff_arr))
            self.strain_jump_threshold = max(
                self.strain_jump_threshold,
                8.0 * diff_mean if diff_mean > 1e-10 else 0.5,
                3.0 * diff_p95 if diff_p95 > 1e-10 else 0.5
            )
        self.strain_jump_threshold = max(0.3, min(1.5, self.strain_jump_threshold))

        # --- AE 斜率比阈值 ---
        if len(self.ae_slope_baseline) > 20:
            slope_arr = np.array(self.ae_slope_baseline)
            slope_mean = float(np.mean(slope_arr))
            slope_std = float(np.std(slope_arr))
            if slope_mean > 1e-12:
                self.ae_slope_ratio_threshold = max(
                    (slope_mean + 5.0 * slope_std) / slope_mean,
                    2.0
                )
            else:
                self.ae_slope_ratio_threshold = 3.0
        else:
            self.ae_slope_ratio_threshold = 3.0
        self.ae_slope_ratio_threshold = max(1.5, min(8.0, self.ae_slope_ratio_threshold))

        # --- FO 漂移阈值 ---
        if len(self.fo_baseline_values) > 10:
            fo_arr = np.array(self.fo_baseline_values)
            fo_std = float(np.std(fo_arr))
            fo_mean = float(np.mean(fo_arr))
            self.baseline_fo_std = fo_std
            if fo_mean > 1e-6:
                fo_cv = fo_std / fo_mean
                self.fo_drift_threshold = max(6.0 * fo_cv, 0.3)
            else:
                self.fo_drift_threshold = 0.5
        else:
            self.fo_drift_threshold = 0.5
        # FO 阈值上限从 1.5 降低到 1.0，因为 Z-score 标准化后漂移通常不超过 1.0 标准差
        self.fo_drift_threshold = max(0.3, min(1.0, self.fo_drift_threshold))

        # --- 自适应冷却周期 ---
        if self.baseline_strain_std is not None:
            base_cv = self.baseline_strain_std / max(abs(strain_mean), 1e-6)
            self.cooldown_period = int(max(300, min(800, 300 + base_cv * 2000)))
        else:
            self.cooldown_period = 500

        # --- 自适应跃迁阈值（更保守，但 Phase 2 阈值降低以允许 AE 尖峰触发） ---
        base_noise_level = self.baseline_strain_std / max(abs(strain_mean), 1e-6) if strain_mean != 0 else 0.1
        noise_factor = 1.0 + min(base_noise_level, 1.0)
        self.transition_thresholds = {
            0: max(0.40, min(0.70, 0.50 * noise_factor)),
            1: max(0.45, min(0.75, 0.55 * noise_factor)),
            2: max(0.40, min(0.70, 0.50 * noise_factor)),  # Phase 2 阈值降低，允许 AE 尖峰触发 Phase 3
        }

    def _detect_strain_jump(self, strain_val, features):
        """检测应变 LEVEL 变化（使用自适应阈值 + 原始应变值）"""
        if strain_val is None or np.isnan(strain_val):
            return 0.0
        self.strain_warm_count += 1
        # ========== 修复5: 延长 warm-up 到 200 点 ==========
        if self.strain_warm_count <= 200:
            if self.strain_baseline_mean is None:
                self.strain_baseline_mean = strain_val
            else:
                self.strain_baseline_mean = 0.995 * self.strain_baseline_mean + 0.005 * strain_val
            return 0.0
        # 方法1: 滑动均值相对基线偏移（使用归一化值）
        if self.strain_buffer.is_warm(50):
            m, _ = self.strain_buffer.get_mean_std(50)
            if m is not None:
                denom = max(abs(self.strain_baseline_mean), 1e-6)
                level_shift = abs(m - self.strain_baseline_mean) / denom
                if level_shift > self.strain_jump_threshold:
                    return min((level_shift - self.strain_jump_threshold) / 2.0, 1.0)
        # 方法2: 使用原始应变值（非归一化值）检测大幅跳变
        raw_original = features.get('strain_raw_original', None)
        if raw_original is not None and not np.isnan(raw_original):
            denom_raw = max(abs(self.strain_baseline_mean), 1e-6)
            raw_shift = abs(raw_original - self.strain_baseline_mean) / denom_raw
            if raw_shift > self.strain_jump_threshold * 3.0:
                return min((raw_shift - self.strain_jump_threshold * 3.0) / 3.0, 1.0)
        return 0.0

    def _detect_ae_transition(self, features):
        """检测AE阶段跃迁：多指标融合（能量加速比 + 事件率突增 + 异常分数 + 原始能量暴增）"""
        ae_conf = 0.0
        # --- 指标1: 累积能量加速比（斜率短窗/长窗比） ---
        # 注意：此指标在 Phase 2 中可能因累积能量饱和而失效，不作为主要判断依据
        if 'ae_cumulative_energy' in features:
            current_energy = features['ae_cumulative_energy']
            slope = current_energy - self.prev_energy if self.prev_energy > 0 else 0.0
            self.prev_energy = current_energy
            self.energy_slopes.append(slope)
            if len(self.energy_slopes) > 500:
                self.energy_slopes.pop(0)
            if len(self.energy_slopes) >= 100:
                short_slopes = self.energy_slopes[-30:]
                long_slopes = self.energy_slopes[-100:]
                short_mean = float(np.mean(short_slopes))
                long_mean = float(np.mean(long_slopes))
                if long_mean > 1e-12:
                    accel_ratio = short_mean / long_mean
                    effective_threshold = max(self.ae_slope_ratio_threshold * 0.6, 1.2)
                    if accel_ratio > effective_threshold:
                        ae_conf = max(ae_conf, min((accel_ratio - effective_threshold) / 2.0, 0.8))
        # --- 指标2: AE 事件率突增（500点后参与，上限降低到 0.3，避免掩盖 spike 检测） ---
        if 'ae_event_rate' in features and self.total_points > 500:
            event_rate = features['ae_event_rate']
            if event_rate > 0.6:
                ae_conf = max(ae_conf, min((event_rate - 0.6) / 0.4, 0.3))
        # --- 指标3: AE 异常分数突增（500点后参与） ---
        if 'ae_anomaly_score' in features and self.total_points > 500:
            score = features['ae_anomaly_score']
            if score > 0.4:
                ae_conf = max(ae_conf, min((score - 0.4) / 0.6, 0.4))
        # ========== 修复6: 指标4 — 原始AE能量尖峰频率检测（主要判断依据） ==========
        # AE原始能量有稀疏但幅度极大的尖峰（比正常高1000-10万倍）
        # 统计每200点窗口内的尖峰次数
        if 'raw_ae_total_energy' in features and self.total_points > 500:
            raw_energy = features['raw_ae_total_energy']
            if not hasattr(self, '_ae_spike_log'):
                self._ae_spike_log = []
            log_e = np.log10(max(raw_energy, 1e-10))
            self._ae_spike_log.append(log_e)
            if len(self._ae_spike_log) > 500:
                self._ae_spike_log.pop(0)
            if len(self._ae_spike_log) >= 200:
                # 基线 = 最近200点的第30百分位数（排除尖峰影响）
                baseline = float(np.percentile(self._ae_spike_log, 30))
                # 尖峰阈值：超过基线 + 1.5 个对数单位（约30倍）
                spike_threshold = baseline + 1.5
                # 统计最近200点中超过阈值的点数
                spikes = sum(1 for v in self._ae_spike_log if v > spike_threshold)
                spike_rate = spikes / len(self._ae_spike_log)
                # 降低触发门槛：spike_rate > 0.03 即开始贡献置信度
                # 提高上限：最高可达 1.0（当 spike_rate >= 0.20 时）
                if spike_rate > 0.03:
                    ae_conf = max(ae_conf, min((spike_rate - 0.03) / 0.17, 1.0))
        return ae_conf

    def _detect_fo_drift(self, features):
        """检测光纤均值偏移（使用自适应阈值 + Z-score 标准化后的 FO）"""
        if 'fo_mean' not in features:
            return 0.0
        self.fo_warm_count += 1
        fo_mean = features['fo_mean']
        # ========== 修复5: 延长 FO warm-up 到 300 点 ==========
        if self.fo_warm_count <= 300:
            if self.fo_baseline_mean is None:
                self.fo_baseline_mean = fo_mean
            else:
                self.fo_baseline_mean = 0.995 * self.fo_baseline_mean + 0.005 * fo_mean
            return 0.0
        # FO 经过 Z-score 标准化，漂移信号以标准差为单位
        # ========== 修复5: 提高 FO 漂移检测阈值 ==========
        if abs(fo_mean) > self.fo_drift_threshold:
            return min((abs(fo_mean) - self.fo_drift_threshold) / 3.0, 1.0)
        # 同时也检测相对基线的偏移比例
        denom = max(abs(self.fo_baseline_mean), 1e-6)
        offset = abs(fo_mean - self.fo_baseline_mean) / denom
        if offset > self.fo_drift_threshold * 1.5:
            return min((offset - self.fo_drift_threshold * 1.5) / 1.0, 1.0)
        return 0.0

    def _decide_transition(self, strain_conf, ae_conf, fo_conf):
        """注意力机制动态权重融合：根据各源当前置信度动态调整权重"""
        self.total_points += 1
        # ========== 修复5: 强制最小稳定期 ==========
        # 前 min_stable_points 点不允许任何跃迁，确保基线充分建立
        if self.total_points <= self.min_stable_points:
            return self.current_phase
        if self.transition_cooldown > 0:
            self.transition_cooldown -= 1
            return self.current_phase
        # 基础权重（阶段先验）
        base_weights = {0: {'strain': 0.4, 'ae': 0.35, 'fo': 0.25},
                        1: {'strain': 0.25, 'ae': 0.50, 'fo': 0.25},
                        2: {'strain': 0.35, 'ae': 0.40, 'fo': 0.25}}
        base_w = base_weights.get(self.current_phase, base_weights[1])
        # ========== 注意力机制：根据各源置信度动态调整权重 ==========
        confs = {'strain': strain_conf, 'ae': ae_conf, 'fo': fo_conf}
        raw_attention = {k: max(v, 0.01) for k, v in confs.items()}
        total_att = sum(raw_attention.values())
        if total_att > 0:
            att_weights = {k: v / total_att for k, v in raw_attention.items()}
            mix_ratio = 0.4
            w = {}
            for k in base_w:
                w[k] = (1 - mix_ratio) * base_w[k] + mix_ratio * att_weights.get(k, 0)
            total_w = sum(w.values())
            if total_w > 0:
                w = {k: v / total_w for k, v in w.items()}
        else:
            w = base_w
        fusion_score = strain_conf * w['strain'] + ae_conf * w['ae'] + fo_conf * w['fo']
        # ========== 改造3: 使用自适应跃迁阈值 ==========
        threshold = self.transition_thresholds.get(self.current_phase, 0.50)
        # ========== 修复5: 每个阶段必须维持至少一定点数才能再次跃迁 ==========
        min_phase_duration = 300  # 每个阶段至少维持 300 点
        points_in_phase = self.total_points - self.phase_entry_points.get(self.current_phase, 0)
        if (fusion_score > threshold and self.current_phase < 3
                and points_in_phase >= min_phase_duration):
            self.current_phase += 1
            self.phase_entry_points[self.current_phase] = self.total_points
            self.transition_cooldown = self.cooldown_period
            print(f'    [阶段跃迁] Phase {self.current_phase - 1} → Phase {self.current_phase} '
                  f'(融合置信度: {fusion_score:.3f}, '
                  f'应变:{strain_conf:.2f} AE:{ae_conf:.2f} FO:{fo_conf:.2f})')
        return self.current_phase

    def update(self, strain_val, features):
        # ========== 改造3: 在 update 中收集基线 ==========
        if not self.baseline_collected:
            self._collect_baseline(strain_val, features)
        if self.transition_cooldown > 0:
            self.transition_cooldown -= 1
        self._update_buffers(strain_val, features)
        strain_conf = self._detect_strain_jump(strain_val, features)
        ae_conf = self._detect_ae_transition(features)
        fo_conf = self._detect_fo_drift(features)
        phase = self._decide_transition(strain_conf, ae_conf, fo_conf)
        self.phase_history.append(phase)
        if len(self.debug_info) < 10000:
            self.debug_info.append({'strain_conf': strain_conf, 'ae_conf': ae_conf,
                                    'fo_conf': fo_conf, 'phase': phase})
        return phase

    def get_stages(self):
        return np.array(self.phase_history, dtype=int)


class OnlineAnomalyDetector:
    """在线异常检测：滑动窗口 Z-score + EWMA"""

    def __init__(self):
        self.strain_buffer = OnlineBuffer([50, 200])
        self.strain_diff_buffer = OnlineBuffer([50, 200])
        self.ae_score_buffer = OnlineBuffer([100, 500])
        self.ae_kurtosis_buffer = OnlineBuffer([100, 500])
        self.fo_max_diff_buffer = OnlineBuffer([50, 200])
        self.fo_zscore_buffers = {}
        self.anomaly_history = {'strain': [], 'ae': [], 'fo': [], 'fused': []}

    def _get_fo_zscore_buffer(self, name):
        if name not in self.fo_zscore_buffers:
            self.fo_zscore_buffers[name] = OnlineBuffer([50, 200])
        return self.fo_zscore_buffers[name]

    def detect_strain(self, strain_val, features):
        if strain_val is None or np.isnan(strain_val):
            return False
        self.strain_buffer.push(strain_val)
        anomaly = False
        if self.strain_buffer.is_warm(50):
            m, s = self.strain_buffer.get_mean_std(50)
            if s is not None and s > 1e-10 and abs(strain_val - m) / s > 3.0:
                anomaly = True
        diff = features.get('strain_diff', 0.0)
        self.strain_diff_buffer.push(diff)
        if self.strain_diff_buffer.is_warm(200):
            p995 = self.strain_diff_buffer.get_percentile(200, 99.5)
            if p995 is not None and diff > p995:
                anomaly = True
        self.anomaly_history['strain'].append(anomaly)
        return anomaly

    def detect_ae(self, features):
        anomaly = False
        if 'ae_anomaly_score' in features:
            score = features['ae_anomaly_score']
            self.ae_score_buffer.push(score)
            if self.ae_score_buffer.is_warm(100):
                m, s = self.ae_score_buffer.get_mean_std(100)
                if s is not None and s > 1e-10 and score > m + 3 * s:
                    anomaly = True
        if 'ae_kurtosis' in features:
            kurt = features['ae_kurtosis']
            self.ae_kurtosis_buffer.push(kurt)
            if self.ae_kurtosis_buffer.is_warm(100):
                p99 = self.ae_kurtosis_buffer.get_percentile(100, 99)
                if p99 is not None and kurt > p99:
                    anomaly = True
        self.anomaly_history['ae'].append(anomaly)
        return anomaly

    def detect_fo(self, features):
        anomaly = False
        if 'fo_range' in features:
            self.fo_max_diff_buffer.push(features['fo_range'])
            if self.fo_max_diff_buffer.is_warm(50):
                p99 = self.fo_max_diff_buffer.get_percentile(50, 99)
                if p99 is not None and features['fo_range'] > p99:
                    anomaly = True
        for key, val in features.items():
            if key.endswith('_zscore') and key.startswith('fo_') and abs(val) > 3.0:
                anomaly = True
        self.anomaly_history['fo'].append(anomaly)
        return anomaly

    def fuse(self, strain_anom, ae_anom, fo_anom, current_phase=1):
        stage_weights = {0: {'strain': 0.3, 'ae': 0.5, 'fo': 0.2},
                         1: {'strain': 0.4, 'ae': 0.4, 'fo': 0.2},
                         2: {'strain': 0.5, 'ae': 0.3, 'fo': 0.2},
                         3: {'strain': 0.6, 'ae': 0.1, 'fo': 0.3}}
        weights = stage_weights.get(current_phase, stage_weights[1])
        available = {}
        if strain_anom is not None: available['strain'] = strain_anom
        if ae_anom is not None: available['ae'] = ae_anom
        if fo_anom is not None: available['fo'] = fo_anom
        if not available:
            fused = False
        else:
            weighted_score = sum(weights[k] * (1.0 if v else 0.0) for k, v in available.items())
            total_w = sum(weights[k] for k in available)
            fused = (weighted_score / total_w) >= 0.5 if total_w > 0 else False
        self.anomaly_history['fused'].append(fused)
        return fused

    def update(self, strain_val, features, current_phase=1):
        strain_anom = self.detect_strain(strain_val, features)
        ae_anom = self.detect_ae(features)
        fo_anom = self.detect_fo(features)
        return self.fuse(strain_anom, ae_anom, fo_anom, current_phase)


# ========== 改造2: StreamSimulator 改为逐块流式读取 ==========
class ChunkedDataReader:
    """逐块数据读取器：不预加载全部数据，按需分块读取

    将已对齐的 DataFrame 保存为临时 CSV 文件，
    然后逐块读取，避免一次性加载全部数据到内存。
    """

    def __init__(self, group_id, chunk_size=1000):
        self.group_id = group_id
        self.chunk_size = chunk_size
        self._temp_file = None
        self._total_rows = 0
        self._reader = None
        self._current_chunk = None
        self._chunk_index = 0
        self._chunk_offset = 0
        self.ae_cols = []
        self.fo_cols = []

    def load_and_prepare(self):
        """加载数据、对齐、保存为临时文件，返回总行数"""
        dl = DataLoader(self.group_id).load_all()
        data = dl.sync_timeline()
        if data is None:
            raise ValueError(f'组 {self.group_id} 数据加载失败')

        self._total_rows = len(data)
        self.ae_cols = [c for c in data.columns if c.startswith('ae_')]
        self.fo_cols = [c for c in data.columns if c.startswith('s') and len(c) <= 3 and c != 'strain']

        # 保存为临时 CSV 文件（不包含归一化，保留原始值）
        self._temp_file = os.path.join(BASE_DIR, f'.temp_{self.group_id}.csv')
        data.to_csv(self._temp_file, index=False)
        print(f'  [ChunkedDataReader] 组 {self.group_id} 已准备: {self._total_rows} 点, '
              f'AE通道={len(self.ae_cols)}, FO通道={len(self.fo_cols)}, '
              f'分块大小={self.chunk_size}')

        # 打开迭代器
        self._reader = pd.read_csv(self._temp_file, chunksize=self.chunk_size)
        self._load_next_chunk()
        return self._total_rows

    def _load_next_chunk(self):
        """加载下一个数据块"""
        try:
            self._current_chunk = next(self._reader)
            self._chunk_index = 0
        except StopIteration:
            self._current_chunk = None

    def read_row(self):
        """读取下一行数据，返回 dict 或 None（已读完）"""
        if self._current_chunk is None:
            return None
        if self._chunk_index >= len(self._current_chunk):
            self._load_next_chunk()
            if self._current_chunk is None:
                return None
        row = self._current_chunk.iloc[self._chunk_index]
        self._chunk_index += 1
        self._chunk_offset += 1
        return row

    def get_row_dict(self, row):
        """将 pandas Series 转为标准 dict 格式"""
        strain_val = float(row.get('strain', 0)) if row.get('strain') is not None else None
        ae_dict = {col: float(row[col]) for col in self.ae_cols if col in row and not np.isnan(row[col])}
        fo_dict = {col: float(row[col]) for col in self.fo_cols if col in row and not np.isnan(row[col])}
        return {'index': self._chunk_offset - 1, 'time': float(row.get('time', self._chunk_offset - 1)),
                'strain': strain_val, 'ae': ae_dict, 'fo': fo_dict}

    def cleanup(self):
        """清理临时文件"""
        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.remove(self._temp_file)
            except (OSError, PermissionError):
                pass

    @property
    def total_rows(self):
        return self._total_rows

    @property
    def current_offset(self):
        return self._chunk_offset


class StreamSimulator:
    """数据流模拟器：逐块流式读取，逐点推送

    【改造2】不再预加载全部数据，改用 ChunkedDataReader 逐块读取。
    """

    def __init__(self, group_id, speed_factor=1.0):
        self.group_id = group_id
        self.speed_factor = speed_factor
        self.reader = ChunkedDataReader(group_id, chunk_size=1000)
        self.total_points = 0
        self.ae_cols = []
        self.fo_cols = []

    def load_data(self):
        n = self.reader.load_and_prepare()
        self.total_points = n
        self.ae_cols = self.reader.ae_cols
        self.fo_cols = self.reader.fo_cols
        return n

    def has_next(self):
        return self.reader.current_offset < self.total_points

    def next_point(self):
        if not self.has_next():
            return None
        row = self.reader.read_row()
        if row is None:
            return None
        return self.reader.get_row_dict(row)

    @property
    def progress(self):
        return self.reader.current_offset / self.total_points if self.total_points > 0 else 0

    def cleanup(self):
        self.reader.cleanup()


class StreamProcessor:
    """在线流式处理主引擎：协调各组件，逐点处理"""

    def __init__(self, group_id, speed_factor=1.0):
        self.group_id = group_id
        self.speed_factor = speed_factor
        self.simulator = StreamSimulator(group_id, speed_factor)
        # ========== 改造1: 集成 OnlineNormalizer ==========
        self.normalizer = OnlineNormalizer(warmup=100)
        self.feature_extractor = OnlineFeatureExtractor()
        self.stage_divider = OnlineStageDivider()
        self.anomaly_detector = OnlineAnomalyDetector()
        self.results = {'time': [], 'strain': [], 'stages': [], 'anomalies': [], 'features': []}

    def load(self):
        return self.simulator.load_data()

    def process_step(self):
        point = self.simulator.next_point()
        if point is None:
            return None
        # ========== 改造1: 先在线归一化，再提取特征 ==========
        # normalize_all 现在返回 (归一化应变, 原始应变, 归一化AE, 归一化FO)
        strain_norm, strain_raw, ae_norm, fo_norm = self.normalizer.normalize_all(
            point['strain'], point['ae'], point['fo'])
        features = self.feature_extractor.extract_all(strain_norm, ae_norm, fo_norm)
        # 将原始应变值存入 features，供阶段划分使用（避免归一化后 raw>0.85 虚高）
        features['strain_raw_original'] = strain_raw if strain_raw is not None else strain_norm
        # 将原始 FO 均值存入 features，供显示使用
        if point['fo']:
            fo_raw_values = [v for v in point['fo'].values() if v is not None]
            if fo_raw_values:
                features['fo_mean_raw'] = float(np.mean(fo_raw_values))
        # ========== 修复5: 将原始 AE 值传入 features，用于能量累积 ==========
        if point['ae']:
            # 计算原始 AE 总能量（所有通道的平方和）
            raw_ae_energy = sum(max(v, 0) ** 2 for v in point['ae'].values() if v is not None)
            features['raw_ae_total_energy'] = raw_ae_energy
        stage = self.stage_divider.update(strain_norm, features)
        anomaly = self.anomaly_detector.update(strain_norm, features, current_phase=stage)
        self.results['time'].append(point['time'])
        self.results['strain'].append(strain_norm)
        self.results['stages'].append(stage)
        self.results['anomalies'].append(anomaly)
        self.results['features'].append(features)
        # 返回更有意义的指标用于显示
        ae_energy_norm = features.get('ae_cumulative_energy_norm', None)
        ae_energy_raw = features.get('ae_cumulative_energy', None)  # 原始累积能量（非归一化）
        fo_mean = features.get('fo_mean', None)  # Z-score 标准化后的 FO 均值
        fo_mean_raw = features.get('fo_mean_raw', None)  # 原始 FO 均值
        ae_spike_rate = None
        if hasattr(self.stage_divider, '_ae_spike_log') and len(self.stage_divider._ae_spike_log) >= 200:
            log_buf = self.stage_divider._ae_spike_log
            baseline = float(np.percentile(log_buf, 30))
            spike_thr = baseline + 1.5
            spikes = sum(1 for v in log_buf if v > spike_thr)
            ae_spike_rate = spikes / len(log_buf)
        return {'time': point['time'], 'index': point['index'], 'strain': strain_norm,
                'stage': stage, 'anomaly': anomaly, 'progress': self.simulator.progress,
                'ae_energy': ae_energy_norm, 'ae_energy_raw': ae_energy_raw,
                'ae_spike_rate': ae_spike_rate,
                'fo_mean': fo_mean, 'fo_mean_raw': fo_mean_raw}

    def run_all(self, callback=None):
        print(f'\n>>> 在线流式处理: 组 {self.group_id}')
        print(f'    数据总量: {self.simulator.total_points} 点')
        print(f'    模拟速度: {self.simulator.speed_factor}x')
        import time
        start_ts = time.time()
        try:
            while self.simulator.has_next():
                result = self.process_step()
                if result is None:
                    break
                if callback:
                    callback(result)
                else:
                    if result['index'] % 1000 == 0:
                        pct = result['progress'] * 100
                        print(f'    进度: {result["index"]}/{self.simulator.total_points} ({pct:.1f}%) '
                              f'阶段={result["stage"]} 异常={result["anomaly"]}')
                delay = 1.0 / self.speed_factor
                time.sleep(delay)
        finally:
            # 确保清理临时文件
            self.simulator.cleanup()
        elapsed = time.time() - start_ts
        print(f'  [完成] 组 {self.group_id} 处理完毕, 耗时 {elapsed:.1f}s')
        return self.get_results()

    def get_results(self):
        return {
            'group_id': self.group_id,
            'time': self.results['time'],
            'strain': self.results['strain'],
            'stages': self.results['stages'],
            'anomalies': self.results['anomalies'],
            'features': self.results['features'],
            'stage_counts': {
                '0': int(np.sum(np.array(self.results['stages']) == 0)),
                '1': int(np.sum(np.array(self.results['stages']) == 1)),
                '2': int(np.sum(np.array(self.results['stages']) == 2)),
                '3': int(np.sum(np.array(self.results['stages']) == 3)),
            },
            'anomaly_count': int(np.sum(self.results['anomalies'])),
            'total_points': len(self.results['time']),
        }


# ============================================================
# 实时可视化仪表盘（Flask + SSE + Chart.js）
# ============================================================

class RealtimeDashboard:
    """实时仪表盘：Flask SSE 推送 + Chart.js 前端渲染"""

    def __init__(self, group_id, speed_factor=10.0, port=5000):
        self.group_id = group_id
        self.speed_factor = speed_factor
        self.port = port
        self.processor = StreamProcessor(group_id, speed_factor)
        self.data_queue = queue.Queue()
        self._running = False

    def _make_sse_app(self):
        app = Flask(__name__)

        @app.route('/')
        def index():
            return HTML_TEMPLATE.replace('{{GROUP_ID}}', self.group_id)

        @app.route('/stream')
        def stream():
            def generate():
                yield f'data: {json.dumps({"type": "meta", "group_id": self.group_id, "speed": self.speed_factor})}\n\n'
                if not self._running:
                    self._running = True
                    t = threading.Thread(target=self._run_processing, daemon=True)
                    t.start()
                while True:
                    try:
                        data = self.data_queue.get(timeout=1)
                        yield f'data: {json.dumps(data)}\n\n'
                    except queue.Empty:
                        yield ': keepalive\n\n'
            return Response(generate(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'Access-Control-Allow-Origin': '*'})

        @app.route('/status')
        def status():
            return jsonify({'running': self._running, 'progress': self.processor.simulator.progress})

        return app

    def _run_processing(self):
        self.processor.load()
        self.processor.run_all(callback=self._on_data)

    def _on_data(self, result):
        self.data_queue.put(result)

    def run(self):
        print(f'\n>>> 启动实时仪表盘: 组 {self.group_id} @ http://localhost:{self.port}')
        print(f'    模拟速度: {self.speed_factor}x')
        app = self._make_sse_app()
        app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


# ============================================================
# HTML 模板（Chart.js 实时图表）
# ============================================================

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>在线结构健康监测 - 组 {{GROUP_ID}}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
h1 { font-size: 1.5rem; margin-bottom: 16px; color: #38bdf8; display: flex; align-items: center; gap: 12px; }
h1 small { font-size: 0.9rem; color: #94a3b8; font-weight: normal; }
.dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1400px; margin: 0 auto; }
.card { background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; }
.card h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 12px; }
.card.full { grid-column: 1 / -1; }
.chart-container { position: relative; height: 200px; }
.info-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.info-item { text-align: center; padding: 12px; background: #0f172a; border-radius: 8px; }
.info-item .label { font-size: 0.75rem; color: #64748b; }
.info-item .value { font-size: 1.5rem; font-weight: bold; color: #38bdf8; }
.info-item .value.phase-0 { color: #22c55e; }
.info-item .value.phase-1 { color: #eab308; }
.info-item .value.phase-2 { color: #f97316; }
.info-item .value.phase-3 { color: #ef4444; }
.log-panel { height: 150px; overflow-y: auto; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.75rem; background: #0f172a; border-radius: 8px; padding: 8px; }
.log-panel div { padding: 2px 0; border-bottom: 1px solid #1e293b; }
.log-panel .phase-transition { color: #fbbf24; font-weight: bold; }
.log-panel .anomaly { color: #ef4444; }
.log-panel .info { color: #94a3b8; }
</style>
</head>
<body>
<h1>🛰️ 在线结构健康监测 <small>组 {{GROUP_ID}} · 实时流式处理</small></h1>
<div class="dashboard">
  <div class="card full">
    <div class="info-grid">
      <div class="info-item"><div class="label">当前阶段</div><div class="value" id="currentPhase">0</div></div>
      <div class="info-item"><div class="label">处理进度</div><div class="value" id="progress">0%</div></div>
      <div class="info-item"><div class="label">异常点数</div><div class="value" id="anomalyCount">0</div></div>
      <div class="info-item"><div class="label">数据点数</div><div class="value" id="pointCount">0</div></div>
    </div>
  </div>
  <div class="card">
    <h2>📈 应变时序</h2>
    <div class="chart-container"><canvas id="strainChart"></canvas></div>
  </div>
  <div class="card">
    <h2>📊 阶段分布</h2>
    <div class="chart-container"><canvas id="stageChart"></canvas></div>
  </div>
  <div class="card">
    <h2>🔊 AE累积能量</h2>
    <div class="chart-container"><canvas id="aeChart"></canvas></div>
  </div>
  <div class="card">
    <h2>🔵 FO均值偏移</h2>
    <div class="chart-container"><canvas id="foChart"></canvas></div>
  </div>
  <div class="card full">
    <h2>📋 运行日志</h2>
    <div class="log-panel" id="logPanel"><div class="info">等待数据...</div></div>
  </div>
</div>
<script>
const groupId = '{{GROUP_ID}}';
const phaseNames = ['健康期', '微损伤萌生', '损伤扩展', '结构失效'];
const phaseColors = ['#22c55e', '#eab308', '#f97316', '#ef4444'];

const strainChart = new Chart(document.getElementById('strainChart'), {
    type: 'scatter',
    data: { datasets: [{ label: '应变', data: [], backgroundColor: '#38bdf8', pointRadius: 2 }] },
    options: { responsive: true, maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: '时间点', color: '#94a3b8' }, grid: { color: '#334155' } },
                 y: { title: { display: true, text: '归一化应变', color: '#94a3b8' }, grid: { color: '#334155' } } },
        plugins: { legend: { labels: { color: '#e2e8f0' } } } }
});

const stageChart = new Chart(document.getElementById('stageChart'), {
    type: 'bar',
    data: { labels: ['健康期', '微损伤萌生', '损伤扩展', '结构失效'],
            datasets: [{ label: '点数', data: [0,0,0,0], backgroundColor: ['#22c55e','#eab308','#f97316','#ef4444'] }] },
    options: { responsive: true, maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, grid: { color: '#334155' } },
                 x: { grid: { display: false } } },
        plugins: { legend: { display: false } } }
});

const aeChart = new Chart(document.getElementById('aeChart'), {
    type: 'line',
    data: { datasets: [{ label: 'AE累积能量(归一化)', data: [], borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.1)', fill: true, pointRadius: 0, borderWidth: 1.5 }] },
    options: { responsive: true, maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: '时间点', color: '#94a3b8' }, grid: { color: '#334155' } },
                 y: { title: { display: true, text: '归一化能量', color: '#94a3b8' }, grid: { color: '#334155' }, min: 0, max: 1 } },
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
        animation: false }
});

const foChart = new Chart(document.getElementById('foChart'), {
    type: 'line',
    data: { datasets: [{ label: 'FO均值', data: [], borderColor: '#22d3ee', backgroundColor: 'rgba(34,211,238,0.1)', fill: true, pointRadius: 0, borderWidth: 1.5 }] },
    options: { responsive: true, maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: '时间点', color: '#94a3b8' }, grid: { color: '#334155' } },
                 y: { title: { display: true, text: '归一化均值', color: '#94a3b8' }, grid: { color: '#334155' } } },
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
        animation: false }
});

const evtSource = new EventSource('/stream');
let pointCount = 0, anomalyCount = 0;
const stageCounts = [0, 0, 0, 0];
const MAX_POINTS = 500;

evtSource.onmessage = function(e) {
    if (e.data.startsWith(':')) return;
    const data = JSON.parse(e.data);
    if (data.type === 'meta') return;
    pointCount++;
    document.getElementById('pointCount').textContent = pointCount;
    document.getElementById('progress').textContent = (data.progress * 100).toFixed(1) + '%';
    if (data.stage !== undefined) {
        const phaseEl = document.getElementById('currentPhase');
        phaseEl.textContent = phaseNames[data.stage] || '未知';
        phaseEl.className = 'value phase-' + data.stage;
        stageCounts[data.stage]++;
        stageChart.data.datasets[0].data = stageCounts;
        stageChart.update();
    }
    if (data.anomaly) {
        anomalyCount++;
        document.getElementById('anomalyCount').textContent = anomalyCount;
    }
    if (data.strain !== undefined && data.strain !== null) {
        const ds = strainChart.data.datasets[0].data;
        ds.push({ x: data.index, y: data.strain });
        if (ds.length > MAX_POINTS) ds.shift();
        strainChart.update('none');
    }
    if (data.ae_energy !== undefined && data.ae_energy !== null) {
        const ds = aeChart.data.datasets[0].data;
        ds.push({ x: data.index, y: data.ae_energy });
        if (ds.length > MAX_POINTS) ds.shift();
        aeChart.update('none');
    }
    if (data.fo_mean !== undefined && data.fo_mean !== null) {
        const ds = foChart.data.datasets[0].data;
        ds.push({ x: data.index, y: data.fo_mean });
        if (ds.length > MAX_POINTS) ds.shift();
        foChart.update('none');
    }
    const log = document.getElementById('logPanel');
    const entry = document.createElement('div');
    if (data.stage !== undefined && data.stage > 0 && pointCount > 1) {
        entry.className = 'phase-transition';
        entry.textContent = `[点${data.index}] 阶段 → ${phaseNames[data.stage]}`;
    } else if (data.anomaly) {
        entry.className = 'anomaly';
        entry.textContent = `[点${data.index}] ⚠ 异常: 应变=${data.strain?.toFixed(4)}`;
    } else {
        entry.className = 'info';
        if (data.index % 500 === 0) {
            entry.textContent = `[点${data.index}] 进度 ${(data.progress*100).toFixed(1)}% · 阶段 ${data.stage}`;
        } else {
            return;
        }
    }
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
};
evtSource.onerror = function() {
    const log = document.getElementById('logPanel');
    const entry = document.createElement('div');
    entry.className = 'anomaly';
    entry.textContent = '[!] 连接断开，尝试重连...';
    log.appendChild(entry);
};
</script>
</body>
</html>'''


# ============================================================
# 入口函数
# ============================================================

def run_online_dashboard(group_id='016', speed_factor=10.0, port=5000):
    """启动实时仪表盘"""
    dashboard = RealtimeDashboard(group_id, speed_factor, port)
    dashboard.run()


def run_online_processing(groups=None, speed_factor=10.0):
    """批量在线处理（无界面）"""
    if groups is None:
        groups = ['016', '017', '018', '019', '020']
    all_results = []
    for gid in groups:
        processor = StreamProcessor(gid, speed_factor=speed_factor)
        processor.load()
        results = processor.run_all()
        all_results.append(results)
        print(f'\n=== 组 {gid} 处理结果 ===')
        print(f'  总点数: {results["total_points"]}')
        print(f'  阶段分布: {results["stage_counts"]}')
        print(f'  异常点数: {results["anomaly_count"]}')
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='在线结构健康监测系统')
    parser.add_argument('mode', nargs='?', default='batch', choices=['dashboard', 'batch'],
                        help='运行模式: dashboard (实时仪表盘) / batch (批量处理)')
    parser.add_argument('--group', default='016', help='仪表盘模式: 组号 (默认 016)')
    parser.add_argument('--groups', nargs='+', default=['016','017','018','019','020'],
                        help='批量模式: 组号列表')
    parser.add_argument('--speed', type=float, default=10.0, help='模拟速度倍率 (默认 10)')
    parser.add_argument('--port', type=int, default=5000, help='仪表盘端口 (默认 5000)')
    args = parser.parse_args()

    if args.mode == 'dashboard':
        run_online_dashboard(args.group, args.speed, args.port)
    else:
        run_online_processing(args.groups, args.speed)


if __name__ == '__main__':
    main()
