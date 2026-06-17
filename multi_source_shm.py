#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞行器结构健康监测 — 多源数据融合处理脚本
融合应变、声发射(AE)、光纤(FO)三种传感器数据，实现：
  1. 故障阶段划分（健康期 → 微损伤萌生 → 损伤扩展 → 结构失效）
  2. 异常点识别（多传感器融合决策）
数据来源：016-019 四组试验数据
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
from scipy.stats import entropy as scipy_entropy
import ruptures as rpt

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SENSOR_WEIGHTS = {
    'healthy':  {'strain': 0.3, 'ae': 0.5, 'fo': 0.2},
    'damage':   {'strain': 0.4, 'ae': 0.4, 'fo': 0.2},
    'failure':  {'strain': 0.5, 'ae': 0.3, 'fo': 0.2},
}
STAGE_NAMES = {0: 'Phase 0: 健康期', 1: 'Phase 1: 微损伤萌生期', 2: 'Phase 2: 损伤扩展期', 3: 'Phase 3: 结构失效期'}
STAGE_COLORS = {0: '#4CAF50', 1: '#FFC107', 2: '#FF9800', 3: '#F44336'}

# ============================================================
# 第一部分：DataLoader
# ============================================================
class DataLoader:
    """数据加载器：统一加载应变、声发射、光纤数据，并进行时间同步和归一化"""

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
        print(f'  [应变] 加载: {os.path.basename(fp)}')
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
        print(f'  [应变] 加载完成: {len(df)} 行, 范围 [{df["strain"].min():.4f}, {df["strain"].max():.4f}]')
        return df

    def load_ae(self):
        fp = os.path.join(self.data_dir, f'{self.group_id}声发射.csv')
        if not os.path.exists(fp):
            print(f'  [警告] 未找到声发射文件: {fp}')
            return None
        print(f'  [AE] 加载: {os.path.basename(fp)}')
        df = pd.read_csv(fp, encoding='utf-8-sig', engine='python')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        self.ae = df
        print(f'  [AE] 加载完成: {len(df)} 行, {len(df.columns)} 特征')
        return df

    def load_fo(self):
        fp = self._find_file([f'{self.group_id}光纤.csv', f'{self.group_id}光纤.xlsx'])
        if fp is None:
            print(f'  [警告] 未找到光纤文件: {self.data_dir}')
            return None
        print(f'  [光纤] 加载: {os.path.basename(fp)}')
        df = pd.read_csv(fp, encoding='utf-8-sig', engine='python') if fp.endswith('.csv') else pd.read_excel(fp)
        df.columns = df.columns.str.strip()
        # 统一列名：时间列→time，光纤通道列→s1/s2/...
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
        print(f'  [光纤] 加载完成: {len(df)} 行, 通道: {[c for c in df.columns if c != "time"]}')
        return df

    def load_all(self):
        print(f'\n{"="*60}\n  加载组 {self.group_id} 数据\n{"="*60}')
        self.load_strain(); self.load_ae(); self.load_fo()
        return self

    def sync_timeline(self):
        if self.strain is None:
            print('  [同步] 无应变数据，跳过')
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
        if self.ae is not None:
            print(f'  [同步] AE插值中 ({len(self.ae.columns)} 列)...')
            ae_time = np.linspace(result['time'].min(), result['time'].max(), len(self.ae))
            target = result['time'].values
            ae_vals = self.ae.values.T
            interp = np.array([np.interp(target, ae_time, col) for col in ae_vals]).T
            for i, col in enumerate(self.ae.columns):
                result[f'ae_{col}'] = interp[:, i]
        self.synced_data = result
        print(f'  [同步] 完成: {len(result)} 行, {len(result.columns)} 列')
        return result

    def normalize(self, data=None):
        data = data if data is not None else self.synced_data
        if data is None:
            return None
        result = data.copy()
        norm_cols = [c for c in result.columns if c != 'time']
        strain_cols = [c for c in norm_cols if c == 'strain']
        ae_cols = [c for c in norm_cols if c.startswith('ae_')]
        fo_cols = [c for c in norm_cols if c.startswith('s') and len(c) <= 3]
        scaler = MinMaxScaler()
        for group, name in [(strain_cols, '应变'), (ae_cols, 'AE'), (fo_cols, '光纤')]:
            if not group:
                continue
            if name == 'AE':
                for col in group:
                    cd = result[col].values.copy()
                    pos = cd > 0
                    if pos.any():
                        cd[pos] = np.log1p(cd[pos])
                    cmin, cmax = cd.min(), cd.max()
                    result[col] = (cd - cmin) / (cmax - cmin) if cmax > cmin else 0
            else:
                result[group] = scaler.fit_transform(result[group].values)
        print(f'  [归一化] 完成: 应变{len(strain_cols)}列, AE{len(ae_cols)}列, 光纤{len(fo_cols)}列')
        return result

# ============================================================
# 第二部分：FeatureExtractor
# ============================================================
class FeatureExtractor:
    """从同步后的多源数据中提取融合特征"""

    def __init__(self, data):
        self.data = data
        self.features = None

    def _get_fo_cols(self):
        return [c for c in self.data.columns if c.startswith('s') and len(c) <= 3 and c != 'strain']

    def extract_strain_features(self):
        if 'strain' not in self.data.columns:
            return pd.DataFrame()
        df = pd.DataFrame(index=self.data.index)
        s = self.data['strain']
        df['strain_raw'] = s
        w = 50
        df['strain_ma'] = s.rolling(w, min_periods=1).mean()
        df['strain_std'] = s.rolling(w, min_periods=1).std()
        df['strain_diff'] = s.diff().abs()
        df['strain_cumdiff'] = df['strain_diff'].cumsum()
        df['strain_deviation'] = (s - df['strain_ma']).abs()
        df['strain_zscore'] = df['strain_deviation'] / df['strain_std'].clip(lower=1e-10)
        return df

    def extract_ae_features(self):
        ae_cols = [c for c in self.data.columns if c.startswith('ae_')]
        if not ae_cols:
            return pd.DataFrame()
        df = pd.DataFrame(index=self.data.index)
        key_map = {'ae_Peak': 'peak', 'ae_Kurtosis': 'kurtosis', 'ae_RMS': 'rms',
                   'ae_SpectralEnergy': 'spectral_energy', 'ae_Mean': 'mean',
                   'ae_Std': 'std', 'ae_Skewness': 'skewness', 'ae_PeakToPeak': 'peak_to_peak'}
        for orig, short in key_map.items():
            if orig in self.data.columns:
                df[f'ae_{short}'] = self.data[orig]
        # --- 新增：滑动窗口高阶统计特征 ---
        w = 100  # 滑动窗口大小
        for feat_name in ['ae_peak', 'ae_rms', 'ae_mean', 'ae_spectral_energy']:
            if feat_name in df.columns:
                s = df[feat_name]
                # 滑动峰度 (Kurtosis)
                kurt = s.rolling(w, min_periods=w // 2).kurt()
                df[f'{feat_name}_kurt'] = kurt.fillna(0)
                # 滑动偏度 (Skewness)
                skew = s.rolling(w, min_periods=w // 2).skew()
                df[f'{feat_name}_skew'] = skew.fillna(0)
                # 滑动脉冲因子 (Impulse Factor = peak / mean of abs)
                ma = s.rolling(w, min_periods=w // 2).mean().clip(lower=1e-10)
                df[f'{feat_name}_impulse'] = (s.abs() / ma).fillna(0)
        # --- 新增：滑动峭度比 (Kurtosis / RMS^2) ---
        if 'ae_kurtosis' in df.columns and 'ae_rms' in df.columns:
            rms_sq = df['ae_rms'].clip(lower=1e-10) ** 2
            df['ae_kurtosis_ratio'] = (df['ae_kurtosis'] / rms_sq).clip(upper=100).fillna(0)
        # --- 原有异常得分 ---
        ws, score = 0, pd.Series(np.zeros(len(self.data)), index=self.data.index)
        for feat, w in [('ae_peak', 0.3), ('ae_kurtosis', 0.3), ('ae_rms', 0.2), ('ae_spectral_energy', 0.2)]:
            if feat in df.columns:
                score += w * df[feat]; ws += w
        if ws > 0:
            df['ae_anomaly_score'] = score / ws
        if 'ae_kurtosis' in df.columns:
            df['ae_event_rate'] = (df['ae_kurtosis'] > 0.5).astype(float).rolling(100, min_periods=1).mean()
        if 'ae_peak' in df.columns:
            df['ae_cumulative_energy'] = (df['ae_peak'] ** 2).cumsum()
            me = df['ae_cumulative_energy'].max()
            if me > 0:
                df['ae_cumulative_energy_norm'] = df['ae_cumulative_energy'] / me
        return df

    def extract_fo_features(self):
        fo_cols = self._get_fo_cols()
        if not fo_cols:
            return pd.DataFrame()
        df = pd.DataFrame(index=self.data.index)
        for col in fo_cols:
            df[f'fo_{col}'] = self.data[col]
        fv = self.data[fo_cols].values
        df['fo_mean'] = np.mean(fv, axis=1)
        df['fo_std'] = np.std(fv, axis=1)
        df['fo_max'] = np.max(fv, axis=1)
        df['fo_min'] = np.min(fv, axis=1)
        df['fo_range'] = df['fo_max'] - df['fo_min']
        w = 50
        for col in fo_cols:
            rm = self.data[col].rolling(w, min_periods=1).mean()
            rs = self.data[col].rolling(w, min_periods=1).std().clip(lower=1e-10)
            df[f'fo_{col}_zscore'] = (self.data[col] - rm) / rs
        df['fo_max_diff'] = df[[f'fo_{c}' for c in fo_cols]].max(axis=1) - df[[f'fo_{c}' for c in fo_cols]].min(axis=1)
        return df

    def extract_all(self):
        frames = [f for f in [self.extract_strain_features(), self.extract_ae_features(), self.extract_fo_features()] if not f.empty]
        self.features = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=self.data.index)
        print(f'  [特征] 提取完成: {len(self.features.columns)} 个特征')
        return self.features

# ============================================================
# 第三部分：StageDivider
# ============================================================
class StageDivider:
    """基于多源数据融合的故障阶段划分"""

    def __init__(self, data, features):
        self.data = data
        self.features = features
        self.stages = self.fused_stages = None

    def _get_fo_cols(self):
        return [c for c in self.data.columns if c.startswith('s') and len(c) <= 3 and c != 'strain']

    def _ensure_monotonic(self, stages):
        for i in range(1, len(stages)):
            stages[i] = max(stages[i], stages[i-1])
        return stages

    def _detect_strain_jump(self):
        """使用百分位阈值检测应变跳变，支持多跳变返回"""
        if 'strain' not in self.data.columns:
            return []
        s = self.data['strain'].values
        sd = np.abs(np.diff(s, prepend=s[0]))
        # 使用 99.5 百分位替代 mean+6σ，自适应于各组数据
        th = np.percentile(sd, 99.5)
        ji = np.where(sd > th)[0]
        if len(ji) == 0:
            return []
        # 对跳变点进行聚类合并（连续索引视为同一跳变事件）
        # 使用位置索引而非值索引进行切片
        jumps = []
        cluster_pos_start = 0
        for i in range(1, len(ji)):
            if ji[i] - ji[i-1] > 10:  # 间隔超过10个点视为不同事件
                # 取簇内最大跳变位置
                cluster_indices = ji[cluster_pos_start:i]  # 使用位置切片
                if len(cluster_indices) > 0:
                    best = cluster_indices[np.argmax(sd[cluster_indices])]
                    jumps.append(best)
                cluster_pos_start = i
        # 处理最后一个簇
        cluster_indices = ji[cluster_pos_start:]
        if len(cluster_indices) > 0:
            best = cluster_indices[np.argmax(sd[cluster_indices])]
            jumps.append(best)
        # 过滤：跳变幅度需超过应变总范围的 10%
        sr = np.percentile(s, 99) - np.percentile(s, 1)
        if sr > 0:
            jumps = [j for j in jumps if sd[j] > sr * 0.1]
        return sorted(jumps)

    def detect_strain_change_points(self):
        if 'strain' not in self.data.columns:
            return np.array([])
        sv = self.data['strain'].values.reshape(-1, 1)
        n = len(sv)
        cps = []
        # 使用改进的多跳变检测
        jumps = self._detect_strain_jump()
        cps.extend(jumps)
        if n > 5000:
            step = max(1, n // 5000)
            si = np.arange(0, n, step)
            sv_s = sv[si]
        else:
            si = np.arange(n)
            sv_s = sv
        try:
            model = rpt.Binseg(model='l2').fit(sv_s)
            for cp in model.predict(pen=max(1, len(sv_s) * 0.05)):
                if 10 < cp < len(sv_s) - 10:
                    oi = si[min(cp, len(si) - 1)]
                    if 10 < oi < n - 10:
                        cps.append(oi)
        except Exception:
            pass
        cps = sorted(set(cps))
        print(f'  [阶段-应变] 检测到 {len(cps)} 个变点: {cps[:5]}...')
        return np.array(cps)

    def detect_ae_stages(self):
        """使用 Binseg 变点检测在累积能量曲线上自适应划分阶段"""
        if 'ae_cumulative_energy_norm' not in self.features.columns:
            return None
        n = len(self.data)
        ce = self.features['ae_cumulative_energy_norm'].values
        stages = np.zeros(n, dtype=int)
        # 在累积能量曲线上检测变点，自适应确定阶段阈值
        ce_2d = ce.reshape(-1, 1)
        n_ce = len(ce_2d)
        if n_ce > 100:
            # 降采样加速
            step = max(1, n_ce // 2000)
            si = np.arange(0, n_ce, step)
            ce_s = ce_2d[si]
            try:
                model = rpt.Binseg(model='l2').fit(ce_s)
                # 检测3个变点对应4个阶段
                cps = model.predict(pen=max(1, len(ce_s) * 0.02))
                cps = sorted([si[min(cp, len(si)-1)] for cp in cps if 5 < cp < len(ce_s) - 5])
            except Exception:
                cps = []
        else:
            cps = []
        if len(cps) >= 3:
            # 使用前3个变点划分4个阶段
            p1, p2, p3 = cps[0], cps[1], cps[2]
            stages[p1:] = np.maximum(stages[p1:], 1)
            stages[p2:] = np.maximum(stages[p2:], 2)
            stages[p3:] = np.maximum(stages[p3:], 3)
        elif len(cps) == 2:
            p1, p2 = cps[0], cps[1]
            stages[p1:] = np.maximum(stages[p1:], 1)
            stages[p2:] = np.maximum(stages[p2:], 2)
        elif len(cps) == 1:
            stages[cps[0]:] = np.maximum(stages[cps[0]:], 1)
        else:
            # 无变点：按能量比例等分
            q33, q66 = np.percentile(ce, 33), np.percentile(ce, 66)
            for i in range(n):
                if ce[i] < q33:
                    stages[i] = 0
                elif ce[i] < q66:
                    stages[i] = 1
                else:
                    stages[i] = 2
        stages = self._ensure_monotonic(stages)
        print(f'  [阶段-AE] 变点数={len(cps)}, 阶段分布: 0={np.sum(stages==0)}, 1={np.sum(stages==1)}, 2={np.sum(stages==2)}, 3={np.sum(stages==3)}')
        return stages

    def detect_fo_stages(self):
        """多统计量融合：均值偏移 + 标准差变化 + 通道间相关性退化"""
        fo_cols = self._get_fo_cols()
        if not fo_cols:
            return None
        n = len(self.data)
        fv = self.data[fo_cols].values
        # 1) 均值偏移比例
        fo_mean = np.mean(fv, axis=1)
        init_m = np.mean(fo_mean[:100])
        denom = max(np.max(np.abs(fo_mean - init_m)), 1)
        mean_offset = np.abs(fo_mean - init_m) / denom
        # 2) 标准差变化比例
        fo_std = np.std(fv, axis=1)
        init_s = np.mean(fo_std[:100])
        denom_s = max(np.max(np.abs(fo_std - init_s)), 1)
        std_offset = np.abs(fo_std - init_s) / denom_s
        # 3) 通道间相关性退化（各通道与均值的偏差一致性）
        channel_dev = np.std(fv - fo_mean.reshape(-1, 1), axis=1)
        init_cd = np.mean(channel_dev[:100])
        denom_cd = max(np.max(np.abs(channel_dev - init_cd)), 1)
        corr_degrad = np.abs(channel_dev - init_cd) / denom_cd
        # 融合得分：均值偏移(0.5) + 标准差变化(0.3) + 相关性退化(0.2)
        fusion_score = 0.5 * mean_offset + 0.3 * std_offset + 0.2 * corr_degrad
        # 自适应阈值：使用百分位
        th_low = np.percentile(fusion_score, 70)
        th_high = np.percentile(fusion_score, 90)
        stages = np.zeros(n, dtype=int)
        for i in range(n):
            if fusion_score[i] < th_low:
                stages[i] = 0
            elif fusion_score[i] < th_high:
                stages[i] = 1
            else:
                stages[i] = 2
        stages = self._ensure_monotonic(stages)
        print(f'  [阶段-光纤] 融合阈值: low={th_low:.3f}, high={th_high:.3f}, '
              f'阶段分布: 0={np.sum(stages==0)}, 1={np.sum(stages==1)}, 2={np.sum(stages==2)}')
        return stages

    def fuse_stages(self):
        n = len(self.data)
        scp = self.detect_strain_change_points()
        ae_s = self.detect_ae_stages()
        fo_s = self.detect_fo_stages()
        # 基础阶段：从 AE 开始
        fused = ae_s.copy() if ae_s is not None else np.zeros(n, dtype=int)
        # 多跳变分段：按跳变点位置将后续阶段提升
        # 将跳变点按位置排序，根据跳变严重程度分配阶段
        if len(scp) > 0:
            # 计算每个跳变点的幅度
            s = self.data['strain'].values if 'strain' in self.data.columns else None
            if s is not None:
                sd = np.abs(np.diff(s, prepend=s[0]))
                # 按幅度排序跳变点
                cp_with_mag = [(cp, sd[cp] if cp < len(sd) else 0) for cp in scp]
                cp_with_mag.sort(key=lambda x: x[1], reverse=True)
                # 最大跳变 -> Phase 3，其余 -> Phase 2
                if len(cp_with_mag) >= 1:
                    top_cp = cp_with_mag[0][0]
                    fused[top_cp:] = np.maximum(fused[top_cp:], 3)
                for mag_cp, _ in cp_with_mag[1:]:
                    fused[mag_cp:] = np.maximum(fused[mag_cp:], 2)
        # 光纤辅助：如果光纤进入 Phase 2+ 但融合阶段仍较低，提升到至少 Phase 1
        if fo_s is not None:
            for i in range(n):
                if fo_s[i] >= 2 and fused[i] < 2:
                    fused[i] = max(fused[i], 1)
        self.fused_stages = self._ensure_monotonic(fused)
        print(f'  [阶段-融合] 最终阶段分布: 0={np.sum(self.fused_stages==0)}, 1={np.sum(self.fused_stages==1)}, 2={np.sum(self.fused_stages==2)}, 3={np.sum(self.fused_stages==3)}')
        return self.fused_stages

# ============================================================
# 第四部分：AnomalyDetector
# ============================================================
class AnomalyDetector:
    """基于多源数据融合的异常点识别"""

    def __init__(self, data, features, stages=None):
        self.data = data
        self.features = features
        self.stages = stages  # 阶段信息，用于阶段感知加权融合
        self.anomalies = {}
        self.fused_anomalies = None

    def _get_fo_cols(self):
        return [c for c in self.data.columns if c.startswith('s') and len(c) <= 3 and c != 'strain']

    def detect_strain_anomaly(self):
        if 'strain' not in self.data.columns:
            return None
        s = self.data['strain'].values
        n = len(s)
        ss = pd.Series(s)
        w = 50
        rm = ss.rolling(w, min_periods=1, center=True).mean()
        rs = ss.rolling(w, min_periods=1, center=True).std().clip(lower=1e-10)
        anomaly = (np.abs(s - rm.values) / rs.values) > 3.0
        sd = np.abs(np.diff(s, prepend=s[0]))
        anomaly |= sd > np.percentile(sd, 99.5)
        print(f'  [异常-应变] 检测到 {np.sum(anomaly)} 个异常点 ({np.sum(anomaly)/n*100:.2f}%)')
        self.anomalies['strain'] = anomaly
        return anomaly

    def detect_ae_anomaly(self):
        ae_cols = [c for c in self.features.columns if c.startswith('ae_')]
        if not ae_cols:
            return None
        n = len(self.data)
        anomaly = np.zeros(n, dtype=bool)
        if 'ae_anomaly_score' in self.features.columns:
            sc = self.features['ae_anomaly_score'].values
            anomaly = sc > (np.mean(sc) + 3 * np.std(sc))
        if 'ae_kurtosis' in self.features.columns:
            anomaly |= self.features['ae_kurtosis'].values > np.percentile(self.features['ae_kurtosis'].values, 99)
        print(f'  [异常-AE] 检测到 {np.sum(anomaly)} 个异常点 ({np.sum(anomaly)/n*100:.2f}%)')
        self.anomalies['ae'] = anomaly
        return anomaly

    def detect_fo_anomaly(self):
        fo_cols = self._get_fo_cols()
        if not fo_cols:
            return None
        n = len(self.data)
        anomaly = np.zeros(n, dtype=bool)
        if 'fo_max_diff' in self.features.columns:
            anomaly |= self.features['fo_max_diff'].values > np.percentile(self.features['fo_max_diff'].values, 99)
        for col in [c for c in self.features.columns if c.endswith('_zscore')]:
            anomaly |= np.abs(self.features[col].values) > 3.0
        print(f'  [异常-光纤] 检测到 {np.sum(anomaly)} 个异常点 ({np.sum(anomaly)/n*100:.2f}%)')
        self.anomalies['fo'] = anomaly
        return anomaly

    def detect_isolation_forest(self):
        """特征分组 + 独立检测：按传感器类型分组训练 Isolation Forest，再融合"""
        n = len(self.data)
        # 定义特征组
        feature_groups = {
            'strain': [c for c in ['strain_raw', 'strain_diff', 'strain_zscore', 'strain_ma', 'strain_std']
                       if c in self.features.columns],
            'ae': [c for c in self.features.columns if c.startswith('ae_') and c != 'ae_cumulative_energy_norm'],
            'fo': [c for c in self.features.columns if c.startswith('fo_')]
        }
        # 如果特征组不足，回退到原有逻辑
        active_groups = {k: v for k, v in feature_groups.items() if len(v) >= 2}
        if not active_groups:
            # 回退：混合所有可用特征
            fcols = []
            for c in ['strain_raw', 'strain_diff', 'strain_zscore']:
                if c in self.features.columns:
                    fcols.append(c)
            if not any(c in self.features.columns for c in ['strain_raw', 'strain_diff']):
                if 'strain' in self.data.columns:
                    self.features['strain_raw'] = self.data['strain']
                    fcols.append('strain_raw')
            for c in ['ae_anomaly_score', 'ae_kurtosis', 'ae_peak']:
                if c in self.features.columns:
                    fcols.append(c)
            for c in ['fo_max_diff', 'fo_std']:
                if c in self.features.columns:
                    fcols.append(c)
            if len(fcols) < 2:
                print('  [异常-IsolationForest] 特征不足，跳过')
                return None
            X = np.nan_to_num(self.features[fcols].values, nan=0.0)
            preds = IsolationForest(contamination=0.05, random_state=42, n_estimators=100).fit_predict(X)
            anomaly_if = preds == -1
        else:
            # 每组独立训练，投票融合
            votes = np.zeros((n, len(active_groups)), dtype=int)
            for idx, (gname, gcols) in enumerate(active_groups.items()):
                Xg = np.nan_to_num(self.features[gcols].values, nan=0.0)
                # 每组使用独立的 contamination 估计
                preds = IsolationForest(contamination=0.05, random_state=42, n_estimators=100).fit_predict(Xg)
                votes[:, idx] = (preds == -1).astype(int)
            # 至少 2 组投票一致视为异常
            vote_sum = np.sum(votes, axis=1)
            anomaly_if = vote_sum >= min(2, len(active_groups))
        print(f'  [异常-IsolationForest] 特征组={list(active_groups.keys())}, '
              f'检测到 {np.sum(anomaly_if)} 个异常点 ({np.sum(anomaly_if)/n*100:.2f}%)')
        self.anomalies['isolation_forest'] = anomaly_if
        return anomaly_if

    def fuse_anomalies(self):
        self.detect_strain_anomaly()
        self.detect_ae_anomaly()
        self.detect_fo_anomaly()
        self.detect_isolation_forest()
        n = len(self.data)
        # 阶段感知权重：不同阶段各传感器的可靠性不同
        # Phase 0 (健康期): AE最敏感(0.5), 应变(0.3), 光纤(0.2)
        # Phase 1 (微损伤): AE(0.4), 应变(0.4), 光纤(0.2)
        # Phase 2 (扩展期): 应变(0.5), AE(0.3), 光纤(0.2)
        # Phase 3 (失效期): 应变(0.6), 光纤(0.3), AE(0.1)
        stage_weights = {
            0: {'strain': 0.3, 'ae': 0.5, 'fo': 0.2},
            1: {'strain': 0.4, 'ae': 0.4, 'fo': 0.2},
            2: {'strain': 0.5, 'ae': 0.3, 'fo': 0.2},
            3: {'strain': 0.6, 'ae': 0.1, 'fo': 0.3},
        }
        # 获取可用传感器列表
        available = [k for k in ['strain', 'ae', 'fo'] if k in self.anomalies and self.anomalies[k] is not None]
        if not available:
            self.fused_anomalies = np.zeros(n, dtype=bool)
            return self.fused_anomalies
        # 计算加权融合得分
        weighted_score = np.zeros(n, dtype=float)
        for i in range(n):
            stage = self.stages[i] if self.stages is not None else 1
            weights = stage_weights.get(stage, stage_weights[1])
            total_w = 0.0
            for k in available:
                weighted_score[i] += weights[k] * self.anomalies[k][i]
                total_w += weights[k]
            if total_w > 0:
                weighted_score[i] /= total_w
        # 阈值：加权得分 >= 0.5 视为异常
        self.fused_anomalies = weighted_score >= 0.5
        # Isolation Forest 辅助：与加权结果共同标记
        if 'isolation_forest' in self.anomalies and self.anomalies['isolation_forest'] is not None:
            if_agree = np.zeros(n, dtype=bool)
            for k in available:
                if_agree |= self.anomalies[k] & self.anomalies['isolation_forest']
            # IF 与至少一个传感器同时标记，且加权得分接近阈值(>=0.3)时也标记
            self.fused_anomalies |= if_agree & (weighted_score >= 0.3)
        print(f'  [异常-融合] 阶段感知加权融合, 最终异常点: {np.sum(self.fused_anomalies)} 个 ({np.sum(self.fused_anomalies)/n*100:.2f}%)')
        return self.fused_anomalies

# ============================================================
# 第五部分：Visualizer
# ============================================================
class Visualizer:
    """生成多源数据融合分析的可视化图表"""

    def __init__(self, group_id, data, features, stages, anomalies):
        self.group_id = group_id
        self.data = data
        self.features = features
        self.stages = stages
        self.anomalies = anomalies
        self.output_dir = os.path.join(OUTPUT_DIR, group_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_fo_cols(self):
        return [c for c in self.data.columns if c.startswith('s') and len(c) <= 3 and c != 'strain']

    def _save(self, fig, name):
        path = os.path.join(self.output_dir, f'{self.group_id}_{name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [可视化] {name}: {path}')

    def _add_stage_background(self, ax, time):
        if self.stages is None:
            return
        bd = [0]
        for i in range(1, len(self.stages)):
            if self.stages[i] != self.stages[i-1]:
                bd.append(i)
        bd.append(len(self.stages) - 1)
        for j in range(len(bd) - 1):
            s, e, st = bd[j], bd[j+1], self.stages[bd[j]]
            ax.axvspan(time[s], time[e], alpha=0.08, color=STAGE_COLORS[st], zorder=-1)

    def plot_timeseries(self):
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        time = self.data['time'].values
        ax = axes[0]
        if 'strain' in self.data.columns:
            ax.plot(time, self.data['strain'].values, 'b-', lw=0.8, label='应变')
            ax.set_ylabel('应变幅值', fontsize=11)
            if self.anomalies is not None and 'strain' in self.anomalies and self.anomalies['strain'] is not None and self.anomalies['strain'].any():
                ax.scatter(time[self.anomalies['strain']], self.data['strain'].values[self.anomalies['strain']],
                           color='red', s=20, marker='x', zorder=5, label='应变异常')
            ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax = axes[1]
        if 'ae_anomaly_score' in self.features.columns:
            ax.plot(time, self.features['ae_anomaly_score'].values, 'orange', lw=0.8, label='AE异常得分')
        if 'ae_cumulative_energy_norm' in self.features.columns:
            ax.plot(time, self.features['ae_cumulative_energy_norm'].values, 'r-', lw=1.0, alpha=0.7, label='AE累积能量(归一化)')
        ax.set_ylabel('AE指标', fontsize=11); ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax = axes[2]
        fo_cols = self._get_fo_cols()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, col in enumerate(fo_cols):
            ax.plot(time, self.data[col].values, color=colors[i % 5], lw=0.8, alpha=0.8, label=col)
        ax.set_xlabel('时间 (s)', fontsize=11); ax.set_ylabel('光纤信号', fontsize=11)
        ax.legend(loc='upper right', ncol=5); ax.grid(True, alpha=0.3)
        if self.stages is not None:
            for ax in axes:
                self._add_stage_background(ax, time)
        fig.suptitle(f'组 {self.group_id} — 多源数据时间序列', fontsize=14, fontweight='bold')
        plt.tight_layout(); self._save(fig, 'timeseries')

    def plot_stages(self):
        if self.stages is None:
            return
        fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
        time = self.data['time'].values
        titles = ['应变 + 融合阶段划分', 'AE累积能量 + 融合阶段划分', '光纤均值 + 融合阶段划分', '融合阶段划分结果']
        data_plots = [
            ('strain', '应变', 'b-'),
            ('ae_cumulative_energy_norm', 'AE累积能量', 'r-'),
            ('fo_mean', '光纤均值', 'g-'),
        ]
        for ax, (key, ylabel, style), title in zip(axes[:3], data_plots, titles[:3]):
            if key in self.data.columns or key in self.features.columns:
                src = self.data if key in self.data.columns else self.features
                ax.plot(time, src[key].values, style, lw=0.8)
            ax.set_ylabel(ylabel, fontsize=11)
            self._add_stage_background(ax, time)
            ax.set_title(title, fontsize=11)
        ax = axes[3]
        for stage in range(4):
            ax.fill_between(time, 0, 1, where=(self.stages == stage),
                            color=STAGE_COLORS[stage], alpha=0.5, label=STAGE_NAMES[stage])
        ax.set_xlabel('时间 (s)', fontsize=11); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.legend(loc='upper right', ncol=4, fontsize=9); ax.set_title(titles[3], fontsize=11)
        fig.suptitle(f'组 {self.group_id} — 故障阶段划分结果', fontsize=14, fontweight='bold')
        plt.tight_layout(); self._save(fig, 'stages')

    def plot_anomalies(self):
        if self.anomalies is None:
            return
        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
        time = self.data['time'].values
        ax = axes[0]
        if 'strain' in self.data.columns:
            sv = self.data['strain'].values
            ax.plot(time, sv, 'b-', lw=0.8, label='应变')
            if 'strain' in self.anomalies and self.anomalies['strain'] is not None:
                ax.scatter(time[self.anomalies['strain']], sv[self.anomalies['strain']],
                           color='red', s=20, marker='x', zorder=5, label='应变异常')
            ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax = axes[1]
        if 'ae_anomaly_score' in self.features.columns:
            ax.plot(time, self.features['ae_anomaly_score'].values, 'orange', lw=0.8, label='AE异常得分')
        if 'ae_cumulative_energy_norm' in self.features.columns:
            ax.plot(time, self.features['ae_cumulative_energy_norm'].values, 'r-', lw=1.0, alpha=0.7, label='AE累积能量(归一化)')
        ax.set_ylabel('AE指标', fontsize=11); ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax = axes[2]
        fo_cols = self._get_fo_cols()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, col in enumerate(fo_cols):
            ax.plot(time, self.data[col].values, color=colors[i % 5], lw=0.8, alpha=0.8, label=col)
        ax.set_xlabel('时间 (s)', fontsize=11); ax.set_ylabel('光纤信号', fontsize=11)
        ax.legend(loc='upper right', ncol=5); ax.grid(True, alpha=0.3)
        if self.stages is not None:
            for ax in axes:
                self._add_stage_background(ax, time)
        fig.suptitle(f'组 {self.group_id} — 多源数据异常点识别结果', fontsize=14, fontweight='bold')
        plt.tight_layout(); self._save(fig, 'anomalies')

# ============================================================
# 第六部分：InformationEvaluator
# ============================================================
class InformationEvaluator:
    """信息论量化评估：熵、互信息、信息保留率、特征重要性、冗余度、融合增益"""

    def __init__(self, group_id, data, features, stages, anomalies):
        self.group_id = group_id
        self.data = data
        self.features = features
        self.stages = stages
        self.anomalies = anomalies
        self.results = {}

    def _get_fo_cols(self):
        return [c for c in self.data.columns if c.startswith('s') and len(c) <= 3 and c != 'strain']

    def _discretize(self, x, bins=20):
        xc = x.copy()
        if np.nanstd(xc) < 1e-10:
            return np.zeros(len(xc), dtype=int)
        xc = np.nan_to_num(xc, nan=0.0)
        xc = (xc - xc.min()) / (xc.max() - xc.min() + 1e-10)
        return np.clip((xc * (bins - 1)).astype(int), 0, bins - 1)

    def calc_entropy(self, x, bins=20):
        d = self._discretize(x, bins)
        _, c = np.unique(d, return_counts=True)
        p = c / c.sum()
        return scipy_entropy(p, base=2)

    def calc_mutual_info(self, x, y, bins=20):
        dx, dy = self._discretize(x, bins), self._discretize(y, bins)
        c = np.zeros((bins, bins))
        for i, j in zip(dx, dy):
            c[i, j] += 1
        c /= c.sum()
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if c[i, j] > 0:
                    pi, pj = c[i, :].sum(), c[:, j].sum()
                    if pi > 0 and pj > 0:
                        mi += c[i, j] * np.log2(c[i, j] / (pi * pj))
        return mi

    def evaluate_source_entropy(self):
        """评估各源数据的熵（信息量）"""
        res = {}
        if 'strain' in self.data.columns:
            res['strain'] = self.calc_entropy(self.data['strain'].values)
        ae_cols = [c for c in self.data.columns if c.startswith('ae_')]
        if ae_cols:
            ae_vals = self.data[ae_cols].values
            res['ae_mean_entropy'] = np.mean([self.calc_entropy(ae_vals[:, i]) for i in range(ae_vals.shape[1])])
            res['ae_total_entropy'] = self.calc_entropy(ae_vals.mean(axis=1))
        fo_cols = self._get_fo_cols()
        if fo_cols:
            fo_vals = self.data[fo_cols].values
            res['fo_mean_entropy'] = np.mean([self.calc_entropy(fo_vals[:, i]) for i in range(fo_vals.shape[1])])
            res['fo_total_entropy'] = self.calc_entropy(fo_vals.mean(axis=1))
        if 'strain' in res and 'ae_total_entropy' in res and 'fo_total_entropy' in res:
            res['fusion_entropy'] = self.calc_entropy(
                np.column_stack([self.data['strain'].values,
                                 ae_vals.mean(axis=1) if ae_cols else np.zeros(len(self.data)),
                                 fo_vals.mean(axis=1) if fo_cols else np.zeros(len(self.data))]).mean(axis=1))
        self.results['source_entropy'] = res
        return res

    def evaluate_mutual_information(self):
        """评估源数据与阶段/异常标签的互信息"""
        res = {}
        if self.stages is not None:
            stage_d = self._discretize(self.stages.astype(float))
            for name, col in [('strain', 'strain')] + \
                             [(f'ae_{i}', c) for i, c in enumerate([c for c in self.data.columns if c.startswith('ae_')])] + \
                             [(f'fo_{c}', c) for c in self._get_fo_cols()]:
                if col in self.data.columns:
                    res[f'{name}_mi_stage'] = self.calc_mutual_info(self.data[col].values, stage_d)
        if self.anomalies is not None and self.anomalies is not None:
            for k in ['strain', 'ae', 'fo']:
                if k in self.anomalies and self.anomalies[k] is not None:
                    for name, col in [('strain', 'strain')] + \
                                     [(f'ae_{i}', c) for i, c in enumerate([c for c in self.data.columns if c.startswith('ae_')])] + \
                                     [(f'fo_{c}', c) for c in self._get_fo_cols()]:
                        if col in self.data.columns:
                            res[f'{name}_mi_anomaly_{k}'] = self.calc_mutual_info(
                                self.data[col].values, self.anomalies[k].astype(float))
        self.results['mutual_information'] = res
        return res

    def evaluate_information_retention_rate(self):
        """评估融合后各源数据的信息保留率 (IRR)"""
        res = {}
        fcols = [c for c in self.features.columns if c != 'time']
        if len(fcols) < 2:
            return res
        fv = np.nan_to_num(self.features[fcols].values, nan=0.0)
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_workers=1).fit(fv)
            ev = pca.explained_variance_ratio_
            res['pca_n_components'] = len(ev)
            res['pca_top1_ratio'] = float(ev[0])
            res['pca_top3_ratio'] = float(ev[:3].sum()) if len(ev) >= 3 else float(ev.sum())
            res['pca_95p_components'] = int(np.searchsorted(ev.cumsum(), 0.95) + 1)
        except Exception:
            pass
        if 'strain' in self.data.columns:
            sv = self.data['strain'].values
            corrs = []
            for col in fcols:
                cv = np.nan_to_num(self.features[col].values, nan=0.0)
                if np.std(sv) > 1e-10 and np.std(cv) > 1e-10:
                    corrs.append(abs(np.corrcoef(sv, cv)[0, 1]))
            res['strain_max_corr'] = max(corrs) if corrs else 0
            res['strain_mean_corr'] = np.mean(corrs) if corrs else 0
        ae_cols = [c for c in self.data.columns if c.startswith('ae_')]
        if ae_cols:
            ae_m = self.data[ae_cols].values.mean(axis=1)
            corrs = []
            for col in fcols:
                cv = np.nan_to_num(self.features[col].values, nan=0.0)
                if np.std(ae_m) > 1e-10 and np.std(cv) > 1e-10:
                    corrs.append(abs(np.corrcoef(ae_m, cv)[0, 1]))
            res['ae_max_corr'] = max(corrs) if corrs else 0
            res['ae_mean_corr'] = np.mean(corrs) if corrs else 0
        fo_cols = self._get_fo_cols()
        if fo_cols:
            fo_m = self.data[fo_cols].values.mean(axis=1)
            corrs = []
            for col in fcols:
                cv = np.nan_to_num(self.features[col].values, nan=0.0)
                if np.std(fo_m) > 1e-10 and np.std(cv) > 1e-10:
                    corrs.append(abs(np.corrcoef(fo_m, cv)[0, 1]))
            res['fo_max_corr'] = max(corrs) if corrs else 0
            res['fo_mean_corr'] = np.mean(corrs) if corrs else 0
        self.results['information_retention'] = res
        return res

    def evaluate_feature_importance(self):
        """评估各特征对阶段划分的贡献度"""
        res = {}
        if self.stages is None:
            return res
        fcols = [c for c in self.features.columns if c != 'time']
        if len(fcols) < 2:
            return res
        X = np.nan_to_num(self.features[fcols].values, nan=0.0)
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
            rf.fit(X, self.stages)
            imp = rf.feature_importances_
            si = np.argsort(imp)[::-1]
            res['top_features'] = [(fcols[i], float(imp[i])) for i in si[:5]]
            res['feature_importances'] = dict(zip(fcols, imp.tolist()))
        except Exception:
            try:
                mi = mutual_info_classif(X, self.stages, random_state=42)
                si = np.argsort(mi)[::-1]
                res['top_features'] = [(fcols[i], float(mi[i])) for i in si[:5]]
                res['feature_importances'] = dict(zip(fcols, mi.tolist()))
            except Exception:
                pass
        self.results['feature_importance'] = res
        return res

    def evaluate_redundancy(self):
        """评估特征间的冗余度"""
        res = {}
        fcols = [c for c in self.features.columns if c != 'time']
        if len(fcols) < 2:
            return res
        fv = np.nan_to_num(self.features[fcols].values, nan=0.0)
        cm = np.corrcoef(fv.T)
        tri = np.triu_indices_from(cm, k=1)
        corrs = cm[tri]
        res['mean_abs_corr'] = float(np.mean(np.abs(corrs)))
        res['max_abs_corr'] = float(np.max(np.abs(corrs)))
        res['high_redundancy_ratio'] = float(np.mean(np.abs(corrs) > 0.8))
        if len(fcols) <= 10:
            pairs = []
            for i in range(len(fcols)):
                for j in range(i + 1, len(fcols)):
                    if abs(cm[i, j]) > 0.8:
                        pairs.append((fcols[i], fcols[j], float(cm[i, j])))
            res['high_redundancy_pairs'] = pairs[:5]
        self.results['redundancy'] = res
        return res

    def evaluate_fusion_gain(self):
        """评估融合相对于单源数据的增益"""
        res = {}
        if self.stages is None:
            return res
        fcols = [c for c in self.features.columns if c != 'time']
        if len(fcols) < 2:
            return res
        X = np.nan_to_num(self.features[fcols].values, nan=0.0)
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
            fusion_score = np.mean(cross_val_score(rf, X, self.stages, cv=3, scoring='f1_macro'))
            res['fusion_f1_score'] = float(fusion_score)
            sensor_groups = {
                'strain_only': [c for c in fcols if c.startswith('strain')],
                'ae_only': [c for c in fcols if c.startswith('ae_')],
                'fo_only': [c for c in fcols if c.startswith('fo_')],
            }
            for name, cols in sensor_groups.items():
                if len(cols) >= 1:
                    Xs = np.nan_to_num(self.features[cols].values, nan=0.0)
                    try:
                        s = np.mean(cross_val_score(rf, Xs, self.stages, cv=3, scoring='f1_macro'))
                        res[f'{name}_f1_score'] = float(s)
                    except Exception:
                        pass
            if 'strain_only_f1_score' in res and 'fusion_f1_score' in res:
                res['strain_gain'] = res['fusion_f1_score'] - res['strain_only_f1_score']
            if 'ae_only_f1_score' in res and 'fusion_f1_score' in res:
                res['ae_gain'] = res['fusion_f1_score'] - res['ae_only_f1_score']
            if 'fo_only_f1_score' in res and 'fusion_f1_score' in res:
                res['fo_gain'] = res['fusion_f1_score'] - res['fo_only_f1_score']
        except Exception:
            pass
        self.results['fusion_gain'] = res
        return res

    def evaluate_all(self):
        self.evaluate_source_entropy()
        self.evaluate_mutual_information()
        self.evaluate_information_retention_rate()
        self.evaluate_feature_importance()
        self.evaluate_redundancy()
        self.evaluate_fusion_gain()
        return self.results

    def print_summary(self):
        r = self.results
        print(f'\n{"="*60}\n  信息论量化评估结果\n{"="*60}')
        if 'source_entropy' in r:
            se = r['source_entropy']
            print(f'\n[信息熵]')
            for k, v in se.items():
                print(f'  {k}: {v:.4f} bits')
        if 'mutual_information' in r:
            mi = r['mutual_information']
            print(f'\n[互信息]')
            top_mi = sorted(mi.items(), key=lambda x: x[1], reverse=True)[:5]
            for k, v in top_mi:
                print(f'  {k}: {v:.4f} bits')
        if 'information_retention' in r:
            ir = r['information_retention']
            print(f'\n[信息保留率]')
            for k in ['pca_top1_ratio', 'pca_top3_ratio', 'pca_95p_components']:
                if k in ir:
                    print(f'  {k}: {ir[k]:.4f}' if isinstance(ir[k], float) else f'  {k}: {ir[k]}')
            for src in ['strain', 'ae', 'fo']:
                for m in ['max_corr', 'mean_corr']:
                    k = f'{src}_{m}'
                    if k in ir:
                        print(f'  {k}: {ir[k]:.4f}')
        if 'feature_importance' in r:
            fi = r['feature_importance']
            print(f'\n[特征重要性 Top5]')
            for name, imp in fi.get('top_features', []):
                print(f'  {name}: {imp:.4f}')
        if 'redundancy' in r:
            rd = r['redundancy']
            print(f'\n[特征冗余度]')
            print(f'  平均绝对相关系数: {rd.get("mean_abs_corr", 0):.4f}')
            print(f'  最大绝对相关系数: {rd.get("max_abs_corr", 0):.4f}')
            print(f'  高冗余比例(>0.8): {rd.get("high_redundancy_ratio", 0):.2%}')
        if 'fusion_gain' in r:
            fg = r['fusion_gain']
            print(f'\n[融合增益]')
            print(f'  融合F1分数: {fg.get("fusion_f1_score", 0):.4f}')
            for src in ['strain_only', 'ae_only', 'fo_only']:
                if f'{src}_f1_score' in fg:
                    print(f'  {src} F1分数: {fg[f"{src}_f1_score"]:.4f}')
            for src in ['strain', 'ae', 'fo']:
                if f'{src}_gain' in fg:
                    print(f'  {src}融合增益: {fg[f"{src}_gain"]:+.4f}')

    def plot_information_matrix(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        r = self.results
        if 'source_entropy' in r:
            se = r['source_entropy']
            names, vals = [], []
            for k, v in se.items():
                names.append(k); vals.append(v)
            axes[0].barh(names, vals, color=['#2196F3', '#FF9800', '#FF5722', '#4CAF50', '#9C27B0'][:len(names)])
            axes[0].set_xlabel('熵 (bits)'); axes[0].set_title('源数据信息熵')
            for i, v in enumerate(vals):
                axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        if 'mutual_information' in r:
            mi = r['mutual_information']
            top = sorted(mi.items(), key=lambda x: x[1], reverse=True)[:8]
            names = [k[:18] + '..' if len(k) > 20 else k for k, _ in top]
            vals = [v for _, v in top]
            axes[1].barh(range(len(vals)), vals, color='#FF9800')
            axes[1].set_yticks(range(len(vals))); axes[1].set_yticklabels(names, fontsize=8)
            axes[1].set_xlabel('互信息 (bits)'); axes[1].set_title('特征-标签互信息 Top8')
        if 'redundancy' in r:
            rd = r['redundancy']
            stats_text = f"平均|r|={rd.get('mean_abs_corr',0):.3f}\n最大|r|={rd.get('max_abs_corr',0):.3f}\n高冗余比例={rd.get('high_redundancy_ratio',0):.1%}"
            axes[2].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=14,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[2].set_title('特征冗余度统计')
            axes[2].axis('off')
        fig.suptitle(f'组 {self.group_id} — 信息论评估矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout(); self._save(fig, 'information_matrix')

    def _save(self, fig, name):
        path = os.path.join(OUTPUT_DIR, self.group_id, f'{self.group_id}_{name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [可视化-信息论] {name}: {path}')

    def plot_entropy_evolution(self):
        """绘制各阶段的信息熵演化"""
        if self.stages is None:
            return
        fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
        time = self.data['time'].values
        ws = 200
        sources = [
            ('strain', '应变', 'b'),
            ('ae_', 'AE', 'orange'),
            ('s', '光纤', 'g'),
        ]
        for ax, (prefix, label, color) in zip(axes, sources):
            cols = [c for c in self.data.columns if c.startswith(prefix) and c != 'strain'] if prefix == 's' else \
                   ([prefix] if prefix == 'strain' else [c for c in self.data.columns if c.startswith(prefix)])
            if not cols:
                ax.text(0.5, 0.5, f'{label} 无数据', ha='center', va='center', transform=ax.transAxes)
                continue
            vals = self.data[cols].values.mean(axis=1)
            ent = np.array([self.calc_entropy(vals[max(0,i-ws):i+ws]) for i in range(len(vals))])
            ax.plot(time, ent, color=color, lw=0.8)
            self._add_stage_background(ax, time)
            ax.set_ylabel(f'{label} 滑动熵', fontsize=11); ax.grid(True, alpha=0.3)
        ax.set_xlabel('时间 (s)', fontsize=11)
        fig.suptitle(f'组 {self.group_id} — 各阶段信息熵演化', fontsize=14, fontweight='bold')
        plt.tight_layout(); self._save(fig, 'entropy_evolution')

    def _add_stage_background(self, ax, time):
        if self.stages is None:
            return
        bd = [0]
        for i in range(1, len(self.stages)):
            if self.stages[i] != self.stages[i-1]:
                bd.append(i)
        bd.append(len(self.stages) - 1)
        for j in range(len(bd) - 1):
            s, e, st = bd[j], bd[j+1], self.stages[bd[j]]
            ax.axvspan(time[s], time[e], alpha=0.08, color=STAGE_COLORS[st], zorder=-1)

    def plot_feature_importance(self):
        """绘制特征重要性条形图"""
        if 'feature_importance' not in self.results:
            return
        fi = self.results['feature_importance']
        if 'top_features' not in fi:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        names = [n for n, _ in fi['top_features']]
        vals = [v for _, v in fi['top_features']]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(vals)))
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('重要性', fontsize=11); ax.set_title(f'组 {self.group_id} — 特征重要性 Top{len(vals)}', fontsize=13)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
        plt.tight_layout(); self._save(fig, 'feature_importance')

    def plot_fusion_matrix(self):
        """绘制融合增益对比图"""
        if 'fusion_gain' not in self.results:
            return
        fg = self.results['fusion_gain']
        fig, ax = plt.subplots(figsize=(10, 6))
        names, vals, colors = [], [], []
        for key in ['fusion_f1_score', 'strain_only_f1_score', 'ae_only_f1_score', 'fo_only_f1_score']:
            if key in fg:
                names.append(key.replace('_f1_score', '').replace('_only', ''))
                vals.append(fg[key])
                colors.append('#4CAF50' if 'fusion' in key else '#2196F3')
        ax.bar(names, vals, color=colors, width=0.5)
        ax.set_ylabel('F1 分数', fontsize=11)
        ax.set_title(f'组 {self.group_id} — 融合 vs 单源 F1 分数对比', fontsize=13)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        ax.set_ylim(0, max(vals) * 1.2 + 0.1)
        plt.tight_layout(); self._save(fig, 'fusion_matrix')

# ============================================================
# 第七部分：主流程
# ============================================================
def process_group(group_id):
    """处理单组数据的完整流程"""
    dl = DataLoader(group_id).load_all()
    data = dl.sync_timeline()
    if data is None:
        return None
    data = dl.normalize(data)
    fe = FeatureExtractor(data)
    features = fe.extract_all()
    sd = StageDivider(data, features)
    stages = sd.fuse_stages()
    ad = AnomalyDetector(data, features, stages=stages)
    anomalies = ad.fuse_anomalies()
    viz = Visualizer(group_id, data, features, stages, ad.anomalies)
    viz.plot_timeseries(); viz.plot_stages(); viz.plot_anomalies()
    ie = InformationEvaluator(group_id, data, features, stages, ad.anomalies)
    ie.evaluate_all(); ie.print_summary()
    ie.plot_information_matrix(); ie.plot_entropy_evolution(); ie.plot_feature_importance(); ie.plot_fusion_matrix()
    return {'group_id': group_id, 'data': data, 'features': features, 'stages': stages,
            'anomalies': ad.anomalies, 'fused_anomalies': ad.fused_anomalies, 'info_eval': ie}

def print_summary(results):
    """打印所有组的汇总结果"""
    print(f'\n{"="*60}')
    print(f'  多源数据融合处理完成 — 汇总报告')
    print(f'{"="*60}')
    rows = []
    for r in results:
        gid = r['group_id']
        stages = r['stages']
        fa = r['fused_anomalies']
        n = len(stages)
        dist = {i: int(np.sum(stages == i)) for i in range(4)}
        ap = int(np.sum(fa)) if fa is not None else 0
        print(f'\n  组 {gid}:')
        print(f'    总样本数: {n}')
        print(f'    阶段分布: 0={dist[0]}({dist[0]/n*100:.1f}%), 1={dist[1]}({dist[1]/n*100:.1f}%), 2={dist[2]}({dist[2]/n*100:.1f}%), 3={dist[3]}({dist[3]/n*100:.1f}%)')
        print(f'    异常点: {ap} ({ap/n*100:.2f}%)')
        rows.append({'组号': gid, '总样本数': n, 'Phase0': dist[0], 'Phase0(%)': f'{dist[0]/n*100:.1f}',
                     'Phase1': dist[1], 'Phase1(%)': f'{dist[1]/n*100:.1f}',
                     'Phase2': dist[2], 'Phase2(%)': f'{dist[2]/n*100:.1f}',
                     'Phase3': dist[3], 'Phase3(%)': f'{dist[3]/n*100:.1f}',
                     '异常点数': ap, '异常率(%)': f'{ap/n*100:.2f}'})
    pdf = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'summary_report.csv')
    pdf.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n  汇总报告已保存: {csv_path}')
    return pdf

def main():
    groups = ['016', '017', '018', '019', '020', '023', '024', '025', '026']
    results = []
    for gid in groups:
        r = process_group(gid)
        if r is not None:
            results.append(r)
    if results:
        print_summary(results)
    print(f'\n{"="*60}\n  所有处理完成！\n{"="*60}')

if __name__ == '__main__':
    main()