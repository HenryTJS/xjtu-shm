# -*- coding: utf-8 -*-
"""评估连续损伤度 D(t)——主样本 6 组（客观 b3 锚，不拿 b1/b2 当裁判）

逐点喂 OnlineDamageIndex（用 StreamSimulator 拿原始应变 + 事件门控 AE 能量）。
输出每组: D 达 0.25/0.55/0.85 的 life% / 相对 b3 提前量 / 断裂前段单调性
结果: results/damage_degree_metrics.csv（缓存 cache/_hi_cache）
"""
import os, sys
sys.path.insert(0, r'd:\lixiang')
os.chdir(r'd:\lixiang')
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex

GROUPS = ['016', '017', '018', '019', '020', '022']   # 主样本 6 组
OUT = r'd:\lixiang\results\damage_degree_metrics.csv'
DNCACHE = r'd:\lixiang\cache\_hi_cache'


def run_one(gid):
    os.makedirs(DNCACHE, exist_ok=True)
    dpath = os.path.join(DNCACHE, f'{gid}.npy')
    if os.path.exists(dpath):
        d = np.load(dpath)
    else:
        di = OnlineDamageIndex()
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
        np.save(dpath, d)
    return gid, d


def _first_ge(d, th):
    idx = np.where(d >= th)[0]
    return float(idx[0]) / len(d) * 100.0 if len(idx) else np.nan


def main():
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(run_one, GROUPS))
    else:
        res = [run_one(g) for g in GROUPS]
    rows = []
    for gid, d in res:
        n = len(d)
        t25, t55, t85 = _first_ge(d, .25), _first_ge(d, .55), _first_ge(d, .85)
        # 断裂前 20% 寿命段的单调性: D 在该段上升的比例
        tail = d[int(n * .80):]
        mono = float(np.mean(np.diff(tail) >= 0)) if len(tail) > 2 else np.nan
        rows.append(dict(gid=gid, n=n, D_end=round(float(d[-1]), 2),
                         t25=round(t25, 1) if not np.isnan(t25) else np.nan,
                         t55=round(t55, 1) if not np.isnan(t55) else np.nan,
                         t85=round(t85, 1) if not np.isnan(t85) else np.nan,
                         lead85=round(99.0 - t85, 1) if not np.isnan(t85) else np.nan,
                         tail_mono=round(mono * 100, 1)))
    df = pd.DataFrame(rows).sort_values('gid')
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    # 汇总
    print('=== D(t) 达阈统计 ===')
    print('  D_end>=0.85 组数:', int((df['D_end'] >= .85).sum()), '/', len(df))
    print('  D_end>=0.55 组数:', int((df['D_end'] >= .55).sum()))
    l85 = df['lead85'].dropna()
    print(f'  达0.85组 平均断裂前提前量={l85.mean():.1f}%  (n={len(l85)})')
    print(df.to_string(index=False))
    print('结果已存:', OUT)


if __name__ == '__main__':
    main()
