#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式读取与在线处理引擎

包含三个核心组件：
- ChunkedDataReader: 逐块数据读取器，不预加载全部数据，按需分块读取
- StreamSimulator:   数据流模拟器，逐块流式读取，逐点推送
- StreamProcessor:   在线流式处理主引擎，协调各组件，逐点处理

【改造2】不再预加载全部数据，改用 ChunkedDataReader 逐块读取。
"""
import os
import time

import numpy as np
import pandas as pd

from .config import BASE_DIR, CHUNK_SIZE
from .data_loader import DataLoader
from .normalizer import OnlineNormalizer
from .feature_extractor import OnlineFeatureExtractor
from .stage_divider import OnlineStageDivider
from .anomaly_detector import OnlineAnomalyDetector


# ========== 改造2: StreamSimulator 改为逐块流式读取 ==========
class ChunkedDataReader:
    """逐块数据读取器：不预加载全部数据，按需分块读取

    将已对齐的 DataFrame 保存为临时 CSV 文件，
    然后逐块读取，避免一次性加载全部数据到内存。
    """

    def __init__(self, group_id, chunk_size=CHUNK_SIZE):
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
        self.reader = ChunkedDataReader(group_id, chunk_size=CHUNK_SIZE)
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
        self.results = {'time': [], 'strain': [], 'stages': [], 'anomalies': []}

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
        # 返回原始值用于可视化显示 — 像 strain 一样，保证每个点都有值
        ae_energy_val = features.get('ae_cumulative_energy', 0.0)  # 始终有值
        fo_mean_val = features.get('fo_mean', 0.0)  # 始终有值（无FO数据时返回最近均值或0）
        ae_spike_rate = None
        if hasattr(self.stage_divider, '_ae_spike_log') and len(self.stage_divider._ae_spike_log) >= 200:
            log_buf = self.stage_divider._ae_spike_log
            baseline = float(np.percentile(log_buf, 30))
            spike_thr = baseline + 1.5
            spikes = sum(1 for v in log_buf if v > spike_thr)
            ae_spike_rate = spikes / len(log_buf)
        return {'time': point['time'], 'index': point['index'], 'strain': strain_norm,
                'stage': stage, 'anomaly': anomaly, 'progress': self.simulator.progress,
                'ae_energy': ae_energy_val,
                'ae_spike_rate': ae_spike_rate,
                'fo_mean': fo_mean_val}  # 直接返回 FO 均值，像 strain 一样

    def run_all(self, callback=None):
        print(f'\n>>> 在线流式处理: 组 {self.group_id}')
        print(f'    数据总量: {self.simulator.total_points} 点')
        print(f'    模拟速度: {self.simulator.speed_factor}x')
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
            'stage_counts': {
                '0': int(np.sum(np.array(self.results['stages']) == 0)),
                '1': int(np.sum(np.array(self.results['stages']) == 1)),
                '2': int(np.sum(np.array(self.results['stages']) == 2)),
                '3': int(np.sum(np.array(self.results['stages']) == 3)),
            },
            'anomaly_count': int(np.sum(self.results['anomalies'])),
            'total_points': len(self.results['time']),
        }
