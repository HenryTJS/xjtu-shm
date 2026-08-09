#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滑动窗口缓冲区：支持多窗口大小，在线计算统计量"""
import numpy as np


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
