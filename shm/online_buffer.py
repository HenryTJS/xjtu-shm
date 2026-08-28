#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滑动窗口缓冲区：环形数组 + 增量统计（优化版）

优化点：
- push O(1)：环形数组覆盖最旧值，避免 list.pop(0) 与 np.array(list) 反复转换
- get_mean_std O(1)：增量维护 sum / sumsq / count（总体标准差，与原 np.std 一致）
- get_percentile / get_min_max 仅在需要时重建数组
"""
import numpy as np


class OnlineBuffer:
    """滑动窗口缓冲区，支持多窗口大小，在线计算统计量"""

    def __init__(self, window_sizes=None):
        if window_sizes is None:
            window_sizes = [50, 100, 200, 500]
        self.windows = {w: np.zeros(w) for w in window_sizes}
        self.heads = {w: 0 for w in window_sizes}
        self.counts = {w: 0 for w in window_sizes}
        self.sums = {w: 0.0 for w in window_sizes}
        self.sumsq = {w: 0.0 for w in window_sizes}

    def push(self, value):
        for w in self.windows:
            arr = self.windows[w]
            head = self.heads[w]
            if self.counts[w] == w:
                # 窗口已满：先移除最旧值（增量更新）
                old = arr[head]
                self.sums[w] -= old
                self.sumsq[w] -= old * old
            else:
                self.counts[w] += 1
            arr[head] = value
            self.sums[w] += value
            self.sumsq[w] += value * value
            self.heads[w] = (head + 1) % w

    def __len__(self, w=None):
        if w is not None:
            return self.counts.get(w, 0)
        return max(self.counts.values()) if self.counts else 0

    def is_warm(self, w, min_ratio=0.5):
        return self.counts.get(w, 0) >= w * min_ratio

    def get_array(self, w):
        """返回窗口内元素（按时间顺序）的 numpy 数组"""
        arr = self.windows[w]
        n = self.counts.get(w, 0)
        if n == 0:
            return np.empty(0)
        if n < w:
            return arr[:n].copy()
        head = self.heads[w]
        if head == 0:
            return arr.copy()
        return np.concatenate([arr[head:], arr[:head]])

    def get_percentile(self, w, q):
        n = self.counts.get(w, 0)
        if n < max(5, w * 0.3):
            return None
        arr = self.get_array(w)
        return float(np.percentile(arr, q))

    def get_mean_std(self, w):
        n = self.counts.get(w, 0)
        if n < 2:
            return None, None
        mean = self.sums[w] / n
        var = max(self.sumsq[w] / n - mean * mean, 0.0)
        return float(mean), float(np.sqrt(var))

    def get_min_max(self, w):
        arr = self.get_array(w)
        if len(arr) == 0:
            return None, None
        return float(np.min(arr)), float(np.max(arr))

    def get_max(self, w=None):
        if w is not None:
            arr = self.get_array(w)
            return float(np.max(arr)) if len(arr) > 0 else 0.0
        w = next(iter(self.windows.keys()))
        arr = self.get_array(w)
        return float(np.max(arr)) if len(arr) > 0 else 0.0
