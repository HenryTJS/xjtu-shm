# -*- coding: utf-8 -*-
"""T1-3 自动弱标签标定 v1（17 组）
协议（可写进论文方法论）：
- 失效锚点 b3 = 数据末端（寿命 ~99%，实验到断才停）
- b2（损伤扩展）= AE 原始 Peak² 累积能量 log 曲线的"最大加速点"（主要能量台阶），限 30%~97%
- b1（损伤启动）= AE 能量首次显著离开早期平台的点（从 b2 向前回溯首个持续增长起点）；若应变先发散则取应变发散点
- 应变辅助：滑动 std 末期发散点若早于 b2 则认为应变主导，用于调整
输出: weak_labels/{sid}_label.csv + weak_labels/labels_summary.csv
"""
import os, sys
sys.path.insert(0, r'd:\lixiang')
os.chdir(r'd:\lixiang')
import numpy as np
import pandas as pd
import io
from shm.data_loader import DataLoader

GROUPS = [f'{i:03d}' for i in list(range(10, 15)) + list(range(16, 28))]


def smooth(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode='same')


def find_b2(logc, grid):
    """能量曲线最大正加速点(主要能量台阶), 限 30%~97% → 返回百分数"""
    d1 = np.gradient(logc, grid)
    d2 = np.gradient(d1, grid)
    mask = (grid > 0.30) & (grid < 0.97)
    if not mask.any():
        return None
    j = int(np.argmax(d2[mask]))
    return float(grid[mask][j] * 100.0)


def find_b1(logc, grid, b2):
    """b1(损伤启动): 累积 log 能量的【水平永久抬升】起点 → 返回百分数

    修订背景：原判据(增长率 d1 超低基线+3σ 且其后 3% 寿命内 >50% 超阈值)会把
    "瞬时 d1 尖峰"误判为萌生——如 020 长静默期在 24% 单点越 3σ 即回落，其累积
    能量水平并未真正抬升 → b1 假阳性。
    修订：b1 = 使累积能量水平较此前平台显著且**持久**抬升（保持到 b2 不回落）的
    最早点：
      pre  = 候选点前 5% 窗 logc 中位（离开前平台水平）
      post = 候选点后 5% 窗 logc 中位（抬升后水平）
      tail = 候选点至 b2 的 logc 中位（抬升是否保持）
    条件：post − pre ≥ 0.4（≈2.5× 能量）且 tail − pre ≥ 0.2。
    平台型(020 静默-爆发) 无此抬升 → 返回 None（phase1 零宽）；台阶型(013)
    在首个大台阶(如 28-30%)处触发。下限 12% 仍保留。
    """
    if b2 is None:
        return None
    lo = 0.12
    hi = (b2 - 1.0) / 100.0
    if hi <= lo:
        return None
    m = (grid >= lo) & (grid <= hi)
    idx = np.where(m)[0]
    d_log = 0.4
    d_tail = 0.2
    # 早期加载/磨合后的平台参考下界：若候选前窗口落在 logc 早期爬升段，pre 取近端即可
    for ik in idx:
        gk = float(grid[ik])
        i0 = int(np.searchsorted(grid, max(lo, gk - 0.05)))
        i1 = max(ik - 1, i0)
        pre = float(np.median(logc[i0:i1 + 1])) if i1 > i0 else float(logc[ik])
        j1 = int(np.searchsorted(grid, min(hi, gk + 0.05)))
        post = float(np.median(logc[ik:j1 + 1])) if j1 > ik else float(logc[ik])
        tail = float(np.percentile(logc[ik:], 50))
        if post - pre >= d_log and tail - pre >= d_tail:
            return gk * 100.0
    return None


def run(gid):
    dl = DataLoader(gid).load_all()
    peak = dl.ae['Peak'].values if dl.ae is not None and 'Peak' in dl.ae.columns else None
    strain = dl.strain['strain'].values if dl.strain is not None else None

    n_ae = len(peak) if peak is not None else 0
    n_s = len(strain) if strain is not None else 0
    if n_ae == 0:
        return None

    # AE 累积能量（0-100% 网格）
    grid = np.linspace(0, 1, 2001)
    e2 = np.maximum(peak, 0.0) ** 2
    cum = np.cumsum(e2)
    logc = np.log10(np.maximum(np.interp(grid, np.linspace(0, 1, n_ae), cum), 1e-12))
    logc = smooth(logc, 21)  # ~1% 平滑

    b2 = find_b2(logc, grid)
    b1 = find_b1(logc, grid, b2) if b2 is not None else None
    b3 = 99.0  # 失效锚点 = 末端

    # 应变末期发散点（辅助 b3 提前 / b1 应变主导）
    strain_diverge = None
    if strain is not None and len(strain) > 200:
        n = len(strain)
        w = max(50, int(n * 0.02))
        sstd = np.convolve(np.abs(np.diff(np.concatenate([[strain[0]], strain]))), np.ones(w)/w, mode='same') if False else None
        # 滑动 std 简易：分块
        nb = 200
        edges = np.linspace(0, n, nb + 1).astype(int)
        blk_std = np.array([float(np.std(strain[edges[i]:edges[i+1]])) if edges[i+1] > edges[i] else 0.0 for i in range(nb)])
        bpct = np.linspace(0, 100, nb)
        mid = (bpct > 25) & (bpct < 60)
        base = float(np.median(blk_std[mid])) if mid.any() else 0.0
        tail = bpct > 50
        if base > 1e-9:
            over = np.where(tail & (blk_std > 1.8 * base))[0]
            if len(over):
                strain_diverge = float(bpct[over[0]])
    # 若应变发散点早于 b2 且 < 95%，说明应变主导，将 b2 取为应变发散点
    if strain_diverge is not None and 0 < strain_diverge < 95:
        if b2 is None or strain_diverge < b2:
            b2 = strain_diverge

    # 组装阶梯 (life% → ref_stage)
    n_grid = 1000  # 0.1% 步长
    life = np.arange(n_grid) * 0.1
    stage = np.zeros(n_grid, dtype=int)
    b1v = b1 if b1 is not None else (b2 if b2 is not None else 60.0)
    b2v = b2 if b2 is not None else (b3 * 0.8)
    stage[life >= b1v] = 1
    stage[life >= b2v] = 2
    stage[life >= b3] = 3

    return dict(gid=gid, b1=round(float(b1v), 1), b2=round(float(b2v), 1), b3=b3,
                n_ae=n_ae, strain_diverge=strain_diverge,
                life=life, stage=stage)


def main():
    all_rows = []
    for gid in GROUPS:
        res = run(gid)
        if res is None:
            print(f'{gid}: 无 AE，跳过')
            continue
        all_rows.append(res)
        # 写单组 csv
        df = pd.DataFrame({'life_pct': res['life'], 'ref_stage': res['stage']})
        df.loc[df['life_pct'] >= res['b3'], 'note'] = 'failure(末端断裂锚点)'
        df.loc[(df['life_pct'] >= res['b2']) & (df['life_pct'] < res['b3']), 'note'] = 'phase2(扩展)'
        df.loc[(df['life_pct'] >= res['b1']) & (df['life_pct'] < res['b2']), 'note'] = 'phase1(微损伤)'
        df.loc[df['life_pct'] < res['b1'], 'note'] = 'phase0(健康/加载)'
        # 抽样 0.5% 存标签（避免过大）
        sel = df.iloc[::2].copy()
        sel.to_csv(rf'd:\lixiang\weak_labels\{gid}_label.csv', index=False, encoding='utf-8-sig')
        print(f"{gid}: b1={res['b1']:6.1f}  b2={res['b2']:6.1f}  b3={res['b3']}  "
              f"strain_diverge={res['strain_diverge']}")

    # 汇总表（弱标签结果只落 csv；说明见 README）
    sumdf = pd.DataFrame([{k: r[k] for k in ('gid', 'b1', 'b2', 'b3', 'n_ae', 'strain_diverge')} for r in all_rows])
    sumdf.to_csv(r'd:\lixiang\weak_labels\labels_summary.csv', index=False, encoding='utf-8-sig')
    print(f'\n共 {len(all_rows)} 组。汇总: weak_labels/labels_summary.csv')


if __name__ == '__main__':
    main()
