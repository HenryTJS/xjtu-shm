# -*- coding: utf-8 -*-
"""T8 消融实验——各证据源/结构对 D(t) 质量与预警的贡献

对 v6 证据结构做 6 组消融(默认=完整 v6)，17 组全量重算 D(t)：
  full        完整 v6: e_ae=max(损伤型,全能量加速); risk=max(e_ae, e_strain); 升快降慢
  no_dmg      去损伤型事件分类  → e_ae=e_full        (检验"事件分类"价值)
  no_full     去全能量加速      → e_ae=e_dmg         (检验"全能量加速"价值, 静默型)
  no_strain   去应变证据        → risk=e_ae          (检验应变贡献/早期污染来源)
  only_strain 仅应变证据        → risk=e_strain      (AE 独立贡献的镜像)
  no_accum    去单调累积        → D=risk(无记忆)     (检验"单调累积"对预警稳定性的作用)
报告: 各模式 达0.55/0.85 组数、A 预警分级、重点组(012/020/017/013/023) D_end 变化。
产物: benchmark/t8_ablation_17.csv + benchmark/t8_report.md + benchmark/figures/t8_ablation.png
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
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, r'd:\lixiang')
os.chdir(r'd:\lixiang')
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex
from eval_common import GROUPS, per_group_metrics

RESULTS = r'd:\lixiang\results'
FIGDIR = r'd:\lixiang\figures'
CACHE = r'd:\lixiang\cache\_t8_cache'
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
MODES = [
    ('full', '完整 v6(基准)'),
    ('no_dmg', '去损伤型事件(仅全能量)'),
    ('no_full', '去全能量加速(仅损伤型)'),
    ('no_strain', '去应变证据(仅 AE)'),
    ('only_strain', '仅应变证据'),
    ('no_accum', '去单调累积(D=risk)'),
]
KEY = ['012', '020', '017', '013', '023']


def run_one(args):
    gid, mode = args
    cdir = os.path.join(CACHE, mode)
    os.makedirs(cdir, exist_ok=True)
    fp = os.path.join(cdir, f'{gid}.npy')
    if os.path.exists(fp):
        return gid, np.load(fp)
    di = OnlineDamageIndex({'abl': mode})
    sim = StreamSimulator(gid)
    sim.load_data()
    dlist = []
    while sim.has_next():
        p = sim.next_point()
        strain = p['strain']
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        dlist.append(di.update(strain, float(pk) if pk is not None else None))
    sim.cleanup()
    d = np.array(dlist, dtype=np.float32)
    np.save(fp, d)
    return gid, d


def main():
    rows_all = []
    agg = []
    workers = 8
    for mode, desc in MODES:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(run_one, [(g, mode) for g in GROUPS]))
        gid_d = dict(res)
        per = {r['gid']: r for r in (per_group_metrics(g, d) for g, d in gid_d.items())}
        df = pd.DataFrame([per[g] for g in GROUPS]).sort_values('gid')
        df.insert(0, 'mode', mode)
        rows_all.append(df)
        n55 = int((df['D_end'] >= .55).sum())
        n85 = int((df['D_end'] >= .85).sum())
        cnt = df['grade'].value_counts()
        agg.append(dict(mode=mode, desc=desc, n55=n55, n85=n85,
                        D_end_med=float(df['D_end'].median()),
                        nA=int(cnt.get('A', 0)), nE=int(cnt.get('E', 0)),
                        nD=int(cnt.get('D', 0)), nC=int(cnt.get('C', 0)),
                        tail_mono=float(df['tail_mono'].mean())))
        print(f'[{desc}] n55={n55} n85={n85} A/E/D/C='
              f'{int(cnt.get("A",0))}/{int(cnt.get("E",0))}/{int(cnt.get("D",0))}/{int(cnt.get("C",0))} '
              f'D_end中位={df["D_end"].median():.2f}', flush=True)
    big = pd.concat(rows_all, ignore_index=True)
    big.to_csv(os.path.join(RESULTS, 'ablation.csv'), index=False, encoding='utf-8-sig')
    # 重点组 D_end 对比
    key_row = big[big['gid'].isin(KEY)].pivot_table(index='gid', columns='mode',
                                                    values='D_end').reindex(KEY)
    print('\n=== 重点组 D_end 对比 ===')
    print(key_row.round(2).to_string())
    # 图: 聚合柱状
    a = pd.DataFrame(agg)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(a))
    axes[0].bar(x, a['n55'], color='tab:blue'); axes[0].set_xticks(x)
    axes[0].set_xticklabels(a['mode'], rotation=30, ha='right'); axes[0].set_title('达 0.55 组数')
    axes[1].bar(x, a['nA'], color='tab:green'); axes[1].set_xticks(x)
    axes[1].set_xticklabels(a['mode'], rotation=30, ha='right'); axes[1].set_title('A-预警精准组数')
    axes[2].bar(x, a['nE'], color='tab:red'); axes[2].set_xticks(x)
    axes[2].set_xticklabels(a['mode'], rotation=30, ha='right'); axes[2].set_title('E-过早组数')
    fig.suptitle('T8 消融: 各证据/结构对 D(t) 的影响')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'ablation.png'), dpi=110)
    # 消融结果聚合已在控制台/CSV 呈现(不再单独生成 md; 结论见 README)
    print(f'结果已存: results/ablation.csv + figures/ablation.png')


if __name__ == '__main__':
    main()
