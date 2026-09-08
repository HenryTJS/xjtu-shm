# -*- coding: utf-8 -*-
"""T10 论文级图
1) 每试件四联图(主样本 6 组):
   A 原始信号(应变块均值 + AE 块能量 log)
   B 各源证据 e_ae / e_strain
   C D(t) + 分级带(0.25/0.55/0.85) + 扩展参考 b2 + 断裂 b3 + 预警点
   D 分级预警级别(0-3 阶梯)
2) 方法流程图 method_flowchart
输出: figures/paper_<gid>.png (×6) + figures/method_flowchart.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, r'd:\lixiang')
os.chdir(r'd:\lixiang')
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex
from eval_common import GROUPS, ref_map

FIG = r'd:\lixiang\figures'
BLK = 500
LV_COLORS = ['#d9ead3', '#fff2cc', '#fce5cd', '#f4cccc']   # 0正常..3临危


def collect(gid):
    """逐点跑 full, 每块记 {pct, strain, ae_en, e_ae, e_strain, D, level}"""
    di = OnlineDamageIndex()
    sim = StreamSimulator(gid)
    sim.load_data()
    pts = 0
    blk_st, blk_en = [], 0.0
    rows = []
    while sim.has_next():
        p = sim.next_point()
        strain = p['strain']
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        if strain is not None and not np.isnan(strain):
            blk_st.append(float(strain))
        if pk is not None:
            blk_en += float(pk) ** 2
        d = di.update(strain, float(pk) if pk is not None else None)
        pts += 1
        if pts % BLK == 0:
            rows.append(dict(
                pct=pts / sim.total_points * 100.0,
                strain=float(np.mean(blk_st)) if blk_st else np.nan,
                ae_en=np.log10(blk_en + 1.0),
                e_ae=di._last_e_ae, e_strain=di._last_e_strain,
                D=float(d), level=di.level))
            blk_st, blk_en = [], 0.0
    sim.cleanup()
    return pd.DataFrame(rows)


def panel_axes(fig):
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.3, 0.6], hspace=0.32,
                          left=0.09, right=0.97, top=0.94, bottom=0.08)
    return [fig.add_subplot(gs[i]) for i in range(4)]


def draw_one(gid):
    df = collect(gid)
    refs = ref_map()
    b2, b3 = refs.get(gid, (np.nan, 99.0))
    fig = plt.figure(figsize=(12, 11))
    _axs = panel_axes(fig)
    ax = fig.axes
    # A 原始信号
    a = ax[0]
    a.plot(df['pct'], df['strain'], color='#1f77b4', lw=0.9, label='应变(块均值)')
    a2 = a.twinx()
    a2.plot(df['pct'], df['ae_en'], color='#d62728', lw=0.8, alpha=0.75, label='AE 块能量 log')
    a2.set_ylabel('log(能量+1)')
    a.set_ylabel('应变')
    a.set_title(f'{gid}  原始多源信号')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left', framealpha=0.6)
    # B 证据
    b = ax[1]
    b.plot(df['pct'], df['e_ae'], color='#9467bd', lw=1.0, label='e_ae (AE 证据)')
    b.plot(df['pct'], df['e_strain'], color='#2ca02c', lw=1.0, label='e_strain (应变证据)')
    b.set_ylim(-0.02, 1.02)
    b.set_ylabel('证据')
    b.legend(fontsize=8, loc='upper left', framealpha=0.6)
    b.set_title('各源损伤证据')
    # C D + 分级
    c = ax[2]
    c.plot(df['pct'], df['D'], color='#000000', lw=1.4, label='D(t)')
    for th, col, lab in [(0.25, '#e07b00', '注意'), (0.55, '#d62728', '预警'), (0.85, '#8b0000', '临危')]:
        c.axhline(th, color=col, lw=0.9, ls='--', alpha=0.6)
        c.text(1.5, th + 0.01, f'{th:.2f} {lab}', color=col, fontsize=7, va='bottom')
    c.axvline(b2, color='#2e7d32', lw=1.0, ls='-.', alpha=0.9)
    c.text(b2 + 0.5, 0.95, 'b2 扩展', color='#2e7d32', fontsize=7, rotation=90, va='top')
    c.axvline(b3, color='k', lw=1.0, ls=':', alpha=0.8)
    c.text(b3 - 0.5, 0.95, 'b3 断裂', fontsize=7, rotation=90, va='top', ha='right')
    # 预警点(不可逆 onset ≥0.3)
    tw = df[df['D'] >= 0.30]
    if len(tw):
        seg = df['D'].to_numpy()
        # 简化: 首次≥0.3 且后续2%窗内未回落到0.15
        idx0 = np.where(seg >= 0.30)[0]
        onset = None
        hold = max(1, int(len(seg) * 0.02))
        for i in idx0:
            if seg[i:min(len(seg), i + hold)].min() >= 0.15:
                onset = df['pct'].iloc[i]
                break
        if onset is not None:
            c.plot(onset, 0.30, 'r*', ms=14, zorder=5)
            c.annotate(f'预警 {onset:.0f}%', (onset, 0.30), textcoords='offset points',
                       xytext=(8, 6), fontsize=8, color='red')
    c.set_ylim(-0.02, 1.02)
    c.set_ylabel('损伤度 D')
    c.legend(fontsize=8, loc='lower right', framealpha=0.6)
    c.set_title('损伤度 D(t) 与分级预警')
    # D 预警级别
    dax = ax[3]
    for lv in range(0, 4):
        m = df['level'] == lv
        dax.fill_between(df['pct'], 0, 1, where=m, color=LV_COLORS[lv], step='post',
                         label=f'级别{lv}({"正常/注意/预警/临危".split("/")[lv]})')
    dax.set_ylim(0, 1)
    dax.set_yticks([])
    dax.set_ylabel('预警级别')
    dax.set_title('分级预警输出(0正常-1注意-2预警-3临危)')
    for a2x in ax:
        a2x.set_xlim(0, 99)
        a2x.grid(alpha=0.3)
    ax[-1].set_xlabel('寿命 (%)')
    out = os.path.join(FIG, f'paper_{gid}.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('已存:', out)


def flowchart():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis('off')
    boxes = [
        (0.03, '原始多源信号\n光纤 / 声发射(AE) / 应变', '#eef3fb'),
        (0.24, '证据提取(块级,在线自适应)\ne_dmg: 损伤型AE事件能量\ne_full: 全能量累积加速度\ne_strain: 应变std发散', '#fdf2e9'),
        (0.47, '证据融合\nrisk = max(e_ae, e_strain)\n(e_ae = max(e_dmg, e_full))', '#f3f0fb'),
        (0.70, '单调累积损伤度\nD(t): 升快降慢(损伤记忆)', '#e9f7ef'),
        (0.90, '分级预警\n0.25注意 / 0.55预警 / 0.85临危', '#fdecec'),
    ]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.35), 0.16, 0.42, boxstyle='round,pad=0.012',
                                    fc=col, ec='0.55', lw=1.1, transform=ax.transAxes))
        ax.text(x + 0.08, 0.56, txt, ha='center', va='center', fontsize=8.5,
                transform=ax.transAxes)
    for x0, x1 in [(0.19, 0.24), (0.40, 0.47), (0.63, 0.70), (0.86, 0.90)]:
        ax.add_patch(FancyArrowPatch((x0 + 0.01, 0.56), (x1 - 0.005, 0.56),
                                     arrowstyle='-|>', mutation_scale=16, lw=1.3,
                                     color='0.3', transform=ax.transAxes))
    ax.text(0.5, 0.92, '多源连续损伤度 D(t) 方法流程（在线因果、零标签）', ha='center',
            fontsize=12, fontweight='bold')
    ax.text(0.5, 0.08, '在线每点更新：证据(块) → risk → D 单调累积 → 分级输出', ha='center',
            fontsize=9, color='0.35')
    out = os.path.join(FIG, 'method_flowchart.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('已存:', out)


def main():
    flowchart()
    for g in GROUPS:
        draw_one(g)


if __name__ == '__main__':
    main()
