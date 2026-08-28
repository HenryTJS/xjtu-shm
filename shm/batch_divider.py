# -*- coding: utf-8 -*-
"""离线批处理阶段划分：作为在线阶段划分的参考基准

基于全局信息（应变水平变点的二分分割），将完整实验划分为 0-3 四阶段。
与在线状态机（因果、逐点）不同，本模块一次性看完全部数据，用作对比基准。
"""
import numpy as np


def cusum_change_point(signal):
    """CUSUM：找单个最大水平变点（返回局部索引）"""
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 4:
        return None
    mu = np.mean(signal)
    cs = np.cumsum(signal - mu)
    return int(np.argmax(np.abs(cs)))


def binary_segmentation(signal, n_points=3, min_seg=50):
    """二分分割：递归找 n_points 个水平变点，返回全局索引列表（升序）"""
    signal = np.asarray(signal, dtype=float)
    points = []
    stack = [(0, len(signal))]
    while len(points) < n_points and stack:
        lo, hi = stack.pop()
        if hi - lo < min_seg * 2:
            continue
        idx = cusum_change_point(signal[lo:hi])
        if idx is None:
            continue
        gidx = lo + idx
        if gidx in points or gidx <= lo or gidx >= hi - 1:
            continue
        points.append(gidx)
        stack.append((lo, gidx))
        stack.append((gidx, hi))
    return sorted(points)


class BatchStageDivider:
    """离线批处理阶段划分基准

    以应变水平变点划分 3 个边界 → 4 阶段（0-3）。
    """

    def __init__(self, n_points=3, smooth_window=200):
        self.n_points = n_points
        self.smooth_window = smooth_window

    def divide(self, strain):
        """strain: 应变序列（一维，长度 = 对齐时间网格行数）
        返回阶段序列（0-3，与输入等长）"""
        strain = np.asarray(strain, dtype=float)
        n = len(strain)
        if n == 0:
            return np.array([], dtype=int)
        smoothed = self._smooth(strain, self.smooth_window)
        pts = binary_segmentation(smoothed, self.n_points)
        stages = np.zeros(n, dtype=int)
        bounds = [0] + pts + [n]
        for i in range(len(bounds) - 1):
            stages[bounds[i]:bounds[i + 1]] = min(i, self.n_points)
        return stages, pts

    @staticmethod
    def _smooth(x, w=200):
        """滑动均值平滑（NaN 用均值填充）"""
        x = np.asarray(x, dtype=float)
        if len(x) < w:
            return x
        x = np.nan_to_num(x, nan=np.nanmean(x))
        k = np.ones(w) / w
        return np.convolve(x, k, mode='same')
