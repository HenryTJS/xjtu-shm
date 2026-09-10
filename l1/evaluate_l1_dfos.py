# -*- coding: utf-8 -*-
"""DFOS(分布式应变/LUNA) 逐块验证 L1-03/04/05
DFOS 每 FBG 块(~5000cycle) 有 ~14 行空间应变快照(0~4924mm 加强筋脚)。
方法(论文式): 比较每块应变分布 vs 健康基准块(前若干块)的偏离 →
  脱粘/失稳使局部应变分布出现峰或整体重分布 → 偏离信号 → HI
输出: 每块偏离指标 + HI + 对齐论文检测点
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.abspath(__file__))   # <项目根>\l1
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

META = {
    'L1-03': dict(n_f=152458, refs=[('冲击后扩展', 10000), ('刚度退化', 69000),
                                    ('AE脱粘', 130000), ('应变脱粘', 143000)]),
    'L1-04': dict(n_f=280098, refs=[('前段低活动', 5000), ('刚度退化(误报)', 30000),
                                    ('应变脱粘', 239500), ('AE脱粘', 260000)]),
    'L1-05': dict(n_f=144969, refs=[('短暂disbond', 66500), ('刚度退化', 68000),
                                    ('AE脱粘', 100000), ('应变脱粘', 110000)]),
}


def load_dfos(gid):
    fp = os.path.join(ROOT, gid, f'{gid}分布式应变.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig')
    t = df['timestamp'].to_numpy(float)
    pos = np.array([float(c[:-2]) for c in df.columns[1:]])
    M = df.iloc[:, 1:].to_numpy(float)
    return t, pos, M


def load_fbg_blocks(gid):
    fp = os.path.join(ROOT, gid, f'{gid}光纤.csv')
    fb = pd.read_csv(fp, encoding='utf-8-sig')
    tf = fb['timestamp'].to_numpy(float)
    gap = np.where(np.diff(tf) > 300.0)[0]
    bounds = np.concatenate([[0], gap + 1, [len(tf)]])
    blk = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 500:
            continue
        blk.append((tf[a], tf[b - 1]))
    return np.array(blk)


def distribute_rows_to_blocks(td, blk):
    """DFOS 行 → FBG 块索引; 返回每行所属块(-1=窗间/后)"""
    n = len(td)
    out = np.full(n, -1)
    for i, (t0, t1) in enumerate(blk):
        m = (td >= t0) & (td <= t1)
        out[m] = i
    return out


def block_profile(M, rows_in_blk):
    """每块: 空间应变分布的代表曲线 = 块内行中位(逐空间点)。"""
    rows = np.where(rows_in_blk)[0]
    if len(rows) == 0:
        return None
    return np.nanmedian(M[rows], axis=0)


def drift_feature(prof, base_prof, pos):
    """块分布 vs 健康基线的偏离。脱粘特征: 局部出现应变峰/凹陷。
    指标: 1) RMSE 全分布差; 2) 局部峰强度(某空间窗内 max|dev|)。
    返回 (rmse, local_peak)。"""
    dev = prof - base_prof
    # 去两端噪声(只在加强筋脚有效区)
    ok = np.isfinite(dev)
    rmse = float(np.sqrt(np.nanmean(dev[ok] ** 2))) if ok.sum() > 10 else np.nan
    # 局部峰: 用滑窗找最强局部偏离(脱粘产生峰)
    w = 100  # ~65mm
    from numpy.lib.stride_tricks import sliding_window_view as sw
    if ok.sum() > 2 * w:
        v = np.where(np.isnan(dev), 0.0, dev)
        sq = sw(np.abs(v), w).mean(axis=1)
        local = float(np.max(sq)) if len(sq) else np.nan
    else:
        local = np.nan
    return rmse, local


def analyze(gid):
    nf = META[gid]['n_f']
    td, pos, M = load_dfos(gid)
    blk = load_fbg_blocks(gid)
    nblk = len(blk)
    blk_cyc = (np.arange(nblk) + 0.5) / nblk * nf
    row_blk = distribute_rows_to_blocks(td, blk)
    # 健康基准块: 前 4 块中位分布(排除块0冲击? 先试前4)
    base_rows = np.isin(row_blk, [0, 1, 2, 3])
    if base_rows.sum() < 5:
        base_rows = row_blk == 0
    base_prof = np.nanmedian(M[base_rows], axis=0)
    # 每块特征
    rmse, local = [], []
    for k in range(nblk):
        prof = block_profile(M, row_blk == k)
        if prof is None:
            rmse.append(np.nan); local.append(np.nan); continue
        r, l = drift_feature(prof, base_prof, pos)
        rmse.append(r); local.append(l)
    rmse = np.array(rmse); local = np.array(local)
    # HI: 用 local(局部峰, 脱粘特征) 归一; 健康期应≈0
    scale = float(np.nanpercentile(local, 90)) if np.nanpercentile(local, 90) > 0 else 1.0
    hi = np.clip(local / scale, 0, 1)
    # 单调累积降噪
    hi2 = np.zeros(nblk)
    for k in range(nblk):
        hi2[k] = max(hi2[k-1], hi[k]) if k > 0 else hi[k]   # 单调包络(保持已达最大)
    hi = hi2
    tail = hi[int(nblk * 0.6):]
    mono = float(np.mean(np.diff(tail) >= 0)) * 100 if len(tail) > 2 else np.nan

    def first_th(th):
        idx = np.where(hi >= th)[0]
        return float(blk_cyc[idx[0]]) if len(idx) else np.nan

    row = dict(gid=gid, n_f=nf, n_blk=nblk, base_prof_nan=int(np.isnan(base_prof).sum()),
               rmse_end=round(float(rmse[-1]), 1), local_scale=round(scale, 1),
               HI_end=round(float(hi[-1]), 3), tail_mono=round(mono, 1),
               warn25=round(first_th(.25), 1) if not np.isnan(first_th(.25)) else np.nan,
               warn50=round(first_th(.5), 1) if not np.isnan(first_th(.5)) else np.nan,
               warn70=round(first_th(.7), 1) if not np.isnan(first_th(.7)) else np.nan)
    print(f'\n[{gid}] 块={nblk}  local_scale={scale:.0f}  HI_end={hi[-1]:.2f}  尾段单调={mono:.0f}%')
    print(f'  HI 达0.25@{row["warn25"]}  0.50@{row["warn50"]}  0.70@{row["warn70"]} (n_f={nf})')
    print('  论文参考: ' + ', '.join(f'{l}@{c}' for l, c in META[gid]['refs']))
    np.savez_compressed(os.path.join(RES, f'_l1_dfos_hi_{gid}.npz'),
                        cyc=blk_cyc, hi=hi, rmse=rmse, local=local,
                        pos=pos, base_prof=base_prof, nf=nf)
    return row


def plot(gid):
    z = np.load(os.path.join(RES, f'_l1_dfos_hi_{gid}.npz'))
    cyc, hi, rmse, local, pos = z['cyc'], z['hi'], z['rmse'], z['local'], z['pos']
    nf = float(z['nf'])
    x = cyc / 1000
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, local, 'o-', color='tab:blue', lw=1.2, label='局部峰偏离')
    axes[0].plot(x, rmse, 's--', color='tab:orange', lw=0.9, label='RMSE')
    axes[0].set_ylabel('DFOS 分布偏离'); axes[0].legend()
    axes[0].set_title(f'{gid}  DFOS(分布式应变) 脱粘特征 (虚线=论文检测, 黑=失效)')
    axes[1].plot(x, hi, 'o-', color='k', lw=1.3)
    for th, c in [(0.25, '#e07b00'), (0.5, '#d62728'), (0.7, '#8b0000')]:
        axes[1].axhline(th, color=c, lw=0.7, ls='--', alpha=0.6)
    axes[1].set_ylabel('HI (局部峰偏离归一)'); axes[1].set_ylim(-0.02, 1.05)
    # 热图: 应变分布随块变化(减基准)
    td, M, blk = None, None, None
    td, _, M = _reload(gid)
    blk = _blk(gid)
    nblk = len(cyc)
    row_blk = _rows(gid, td, blk)
    base_prof = z['base_prof']
    hm = np.full((nblk, M.shape[1]), np.nan)
    for k in range(nblk):
        rows = np.where(row_blk == k)[0]
        if len(rows):
            hm[k] = np.nanmedian(M[rows], axis=0) - base_prof
    im = axes[2].imshow(hm, aspect='auto', origin='lower',
                        extent=[pos[0], pos[-1], 0, nblk], cmap='RdBu_r',
                        vmin=-np.nanpercentile(np.abs(hm), 99), vmax=np.nanpercentile(np.abs(hm), 99))
    axes[2].set_ylabel('块 #'); axes[2].set_xlabel('沿加强筋脚位置 (mm)')
    axes[2].set_title('应变分布 vs 基准 (红=拉应变峰)')
    fig.colorbar(im, ax=axes[2])
    for ax in axes[:2]:
        for lab, c in META[gid]['refs']:
            ax.axvline(c / 1000, color='grey', ls=':', lw=0.8)
        ax.axvline(nf / 1000, color='k', ls='--', lw=1)
    fig.tight_layout()
    out = os.path.join(FIG, f'l1_dfos_{gid}.png')
    fig.savefig(out, dpi=110)
    print('已存:', out)


_cache = {}


def _reload(gid):
    if gid not in _cache:
        _cache[gid] = load_dfos(gid)
    return _cache[gid]


def _blk(gid):
    if ('blk', gid) not in _cache:
        _cache[('blk', gid)] = load_fbg_blocks(gid)
    return _cache[('blk', gid)]


def _rows(gid, td, blk):
    if ('rows', gid) not in _cache:
        _cache[('rows', gid)] = distribute_rows_to_blocks(td, blk)
    return _cache[('rows', gid)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(META))
    ap.add_argument('--mode', default='hi', choices=['hi', 'plot'])
    a = ap.parse_args()
    groups = a.groups.split(',')
    if a.mode == 'hi':
        rows = [analyze(g) for g in groups]
        pd.DataFrame(rows).to_csv(os.path.join(RES, 'l1_dfos_metrics.csv'),
                                  index=False, encoding='utf-8-sig')
        for g in groups:
            plot(g)
    else:
        for g in groups:
            plot(g)


if __name__ == '__main__':
    main()
