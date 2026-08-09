#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线特征提取：维护滑动窗口状态，逐点计算特征

对三种传感器（应变、声发射 AE、光纤 FO）逐点提取特征：
- 应变: 滑动均值、标准差、差分、累积差分、Z-score
- AE:   峰值、峰度、RMS、谱能量、累积能量、事件率、异常分数
- FO:   各通道均值/标准差/极差、通道偏移、Z-score

所有特征均基于历史滑动窗口计算，不窥探未来。
"""
import numpy as np
import pandas as pd

from .online_buffer import OnlineBuffer


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
        # 始终返回累积能量（即使本次无AE数据，也保持上一个值）
        feats['ae_cumulative_energy'] = self.ae_energy_accum
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
            # 即使无FO数据，也返回一个默认值，保证图表连续性
            if hasattr(self, 'fo_mean_buffer') and len(self.fo_mean_buffer.get_array(50)) > 0:
                feats['fo_mean'] = float(np.mean(self.fo_mean_buffer.get_array(50)))
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
