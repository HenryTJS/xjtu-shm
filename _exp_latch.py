# -*- coding: utf-8 -*-
"""方向A 对照实验(临时): 损伤确认后加速追赶(latch_enable) 对 0.85 可达性与误报的影响
对比: off(v6 默认) vs on(latch)。6 组主样本。
输出: 达 0.85 组数 / A-E-D-C 分级 / D_end / 逐组对照(是否引入早期误报)
"""
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, r'd:\lixiang')
os.chdir(r'd:\lixiang')
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex
from eval_common import GROUPS, per_group_metrics

CACHE = r'd:\lixiang\cache\_latch_cache'


def run_one(args):
    gid, latch = args
    mode = 'on' if latch else 'off'
    cdir = os.path.join(CACHE, mode)
    os.makedirs(cdir, exist_ok=True)
    fp = os.path.join(cdir, f'{gid}.npy')
    if os.path.exists(fp):
        return gid, mode, np.load(fp)
    di = OnlineDamageIndex({'latch_enable': latch})
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
    return gid, mode, d


def main():
    jobs = [(g, l) for g in GROUPS for l in (False, True)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run_one, jobs))
    per = {m: {} for m in ('off', 'on')}
    for gid, mode, d in res:
        per[mode][gid] = per_group_metrics(gid, d)
    rows = []
    for g in GROUPS:
        o, n = per['off'][g], per['on'][g]
        rows.append(dict(gid=g,
                         D_end_off=o['D_end'], D_end_on=n['D_end'],
                         t85_off=o['t85'], t85_on=n['t85'],
                         t_warn_off=o['t_warn'], t_warn_on=n['t_warn'],
                         grade_off=o['grade'], grade_on=n['grade'],
                         b2=o['b2']))
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 160)
    print(df.round(2).to_string(index=False))
    n85o = int((df['D_end_off'] >= .85).sum())
    n85n = int((df['D_end_on'] >= .85).sum())
    n55o = int((df['D_end_off'] >= .55).sum())
    n55n = int((df['D_end_on'] >= .55).sum())
    go = df['grade_off'].value_counts().to_dict()
    gn = df['grade_on'].value_counts().to_dict()
    print('\n=== 汇总 ===')
    print(f'达0.55: off={n55o}  on={n55n}')
    print(f'达0.85: off={n85o}  on={n85n}')
    print(f'A/E/D/C off: {go}')
    print(f'A/E/D/C on : {gn}')
    # 误报检查: grade 从 A 变 E/D(early/late)
    df['regress'] = (df['grade_off'] == 'A') & (df['grade_on'] != 'A')
    if df['regress'].any():
        print('\n⚠️ 开启 latch 后变差的组:', df[df['regress']]['gid'].tolist())
    else:
        print('\n无组因 latch 从 A 变差(未引入误报)')
    df.to_csv(r'd:\lixiang\results\_latch_exp.csv', index=False, encoding='utf-8-sig')
    print('已存: results/_latch_exp.csv')


if __name__ == '__main__':
    main()
