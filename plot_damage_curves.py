# -*- coding: utf-8 -*-
"""T6 配套: 17 组 D(t) 曲线总览图（含分级带 + b2/b3 参考线）

读 _t7_cache 的 default 配置(与 hi_metrics v6 同参)逐点 D。
输出: benchmark/figures/t6_D_curves_17.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, r'd:\lixiang')
os.chdir(r'd:\lixiang')
from eval_common import GROUPS, ref_map

CACHE = r'd:\lixiang\cache\_hi_cache'      # 逐点 D 缓存(由 evaluate_damage_degree 生成)
OUTD = r'd:\lixiang\figures'
os.makedirs(OUTD, exist_ok=True)
refs = ref_map()


def main():
    fig, axes = plt.subplots(5, 4, figsize=(18, 20))
    axes = axes.ravel()
    for ax, g in zip(axes, GROUPS):
        d = np.load(os.path.join(CACHE, f'{g}.npy'))
        x = np.linspace(0, 100, len(d))
        ax.plot(x, d, lw=0.8, color='tab:blue')
        for th, c in [(0.25, 'tab:orange'), (0.55, 'tab:red'), (0.85, 'darkred')]:
            ax.axhline(th, color=c, lw=0.6, ls='--', alpha=0.6)
        b2, b3 = refs.get(g, (np.nan, 99.0))
        ax.axvline(b2, color='tab:green', lw=0.8, ls='-.', alpha=0.8)
        ax.axvline(b3, color='k', lw=0.8, ls=':', alpha=0.8)
        ax.set_title(f'{g}  D_end={d[-1]:.2f}', fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 25, 50, 75, 99])
        ax.grid(alpha=0.3)
    axes[-1].axis('off')
    fig.suptitle('T6 D(t) 曲线（默认 v6；分级 0.25/0.55/0.85；绿虚线 b2，黑点线 b3=断裂）',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUTD, 'damage_degree_curves.png')
    fig.savefig(out, dpi=110)
    print('已存:', out)


if __name__ == '__main__':
    main()
