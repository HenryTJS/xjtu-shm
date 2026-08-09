#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线异常检测：滑动窗口 Z-score + EWMA

对三种传感器（应变、声发射 AE、光纤 FO）分别检测异常，
再根据当前阶段权重融合为综合异常判定。
"""
import numpy as np

from .online_buffer import OnlineBuffer


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
