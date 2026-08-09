#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线滑动窗口归一化：仅依赖历史数据，不窥探未来

对每种传感器类型维护独立的滑动窗口缓冲区：
- 应变: 使用 OnlineBuffer 滚动 min/max 进行 min-max 归一化
- AE:   先 log1p 变换，再滚动 min-max 归一化
- FO:   使用 Z-score 标准化（保留漂移信息），而非 min-max 压缩

在 warm-up 阶段（窗口未满）返回原始值，避免早期不稳定。
"""
import numpy as np

from .online_buffer import OnlineBuffer
from .config import NORMALIZER_WARMUP, NORMALIZER_WINDOW


class OnlineNormalizer:
    """在线滑动窗口归一化：仅依赖历史数据，不窥探未来"""

    def __init__(self, warmup=NORMALIZER_WARMUP):
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
        if self.count < self.warmup or not self.strain_buffer.is_warm(NORMALIZER_WINDOW):
            return raw, raw  # warm-up 阶段返回原始值
        cmin, cmax = self.strain_buffer.get_min_max(NORMALIZER_WINDOW)
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
            if self.count < self.warmup or not buf.is_warm(NORMALIZER_WINDOW):
                result[key] = cd  # warm-up 返回 log1p 值
            else:
                cmin, cmax = buf.get_min_max(NORMALIZER_WINDOW)
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
            if self.count < self.warmup or not buf.is_warm(NORMALIZER_WINDOW):
                result[key] = val  # warm-up 返回原始值
            else:
                # Z-score 标准化：保留漂移方向和幅度
                m, s = buf.get_mean_std(NORMALIZER_WINDOW)
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
