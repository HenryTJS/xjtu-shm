# -*- coding: utf-8 -*-
"""T7 Leave-One-Specimen-Out (LOSO) 参数选择 + 泛化评估

前提: 已运行 t7_sens.py(缓存 _t7_cache 含 default + 每参数 ±30% 两档)。
LOSO 逐参数进行: 候选档 C_p = {default, p×0.7, p×1.3}(其余参数默认)。
  对每个留出组 g: 在其余 16 组上按聚合目标选最优档 c*(g);
  评估: 留出组在 c*(g) 下的表现 vs 默认档。
报告: 各参数"选档分布"——若多数折选 default → 参数不敏感/LOSO 稳定;
      否则指出敏感参数及其留出泛化代价。

注意: b2/b3 仅用于离线选参目标(与 T3/T4 口径一致), 不进入 D 的在线计算。
运行: python leave_one_out_cv.py
输出: results/leave_one_out_cv.csv + 控制台总结
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (PARS, DEFAULT, GROUPS, cfg_id, T7CACHE,
                         per_group_metrics)

OUT = os.path.join(r'd:\lixiang\results', 'leave_one_out_cv.csv')


def load_table(params):
    """从缓存读 {gid:d} 并转逐组指标表。缺缓存则报错提示先跑 t7_sens。"""
    cid = cfg_id(params)
    cdir = os.path.join(T7CACHE, cid)
    rows = []
    for g in GROUPS:
        fp = os.path.join(cdir, f'{g}.npy')
        if not os.path.exists(fp):
            sys.exit(f'缺缓存 {fp} —— 请先运行 t7_sens.py(或单独补算该配置)')
        d = np.load(fp)
        rows.append(per_group_metrics(g, d))
    return pd.DataFrame(rows)


def cand_cfgs(p):
    """参数 p 的 3 档候选(其余默认)。"""
    out = []
    for m in (0.7, 1.0, 1.3):
        cp = dict(DEFAULT)
        cp[p] = DEFAULT[p] * m
        out.append((m, cp))
    return out


def score(df):
    """聚合目标(越大越好): A 对齐 + 达级 + 漏报重罚。"""
    nA = int((df['grade'] == 'A').sum())
    n55 = int((df['D_end'] >= .55).sum())
    nC = int((df['grade'] == 'C').sum())
    return nA + 0.25 * n55 - 1.5 * nC


def main():
    recs = []
    for p in PARS:
        tabs = {m: load_table(cp) for m, cp in cand_cfgs(p)}
        tab0 = tabs[1.0]
        chosen = []
        diff_rows = []
        for g in GROUPS:
            train = [x for x in GROUPS if x != g]
            best_m, best_s = None, -1e18
            for m, cp in cand_cfgs(p):
                s = score(tabs[m][tabs[m]['gid'].isin(train)])
                if s > best_s:
                    best_s, best_m = s, m
            chosen.append(best_m)
            # 留出组表现: 默认档 vs 被选档
            row0 = tab0[tab0['gid'] == g].iloc[0]
            rowc = tabs[best_m][tabs[best_m]['gid'] == g].iloc[0]
            diff_rows.append(dict(gid=g, par=p, chosen_m=best_m,
                                  g0=row0['grade'], gc=rowc['grade'],
                                  err0=row0['err'], errc=rowc['err'],
                                  dend0=row0['D_end'], dendc=rowc['D_end']))
        ch = pd.Series(chosen).value_counts().to_dict()
        diff = pd.DataFrame(diff_rows)
        improved = int(((diff['g0'] == 'E') & (diff['gc'] == 'A')).sum())
        worsened = int(((diff['g0'].isin(['A', 'D'])) & (diff['gc'] == 'E')).sum())
        nA0 = int((tab0['grade'] == 'A').sum())
        recs.append(dict(par=p, desc='', ch_0_7=ch.get(0.7, 0),
                         ch_1_0=ch.get(1.0, 0), ch_1_3=ch.get(1.3, 0),
                         nA_default=nA0, EtoA=improved, AtoE=worsened))
        print(f'[{p}] 选档(0.7/1.0/1.3)={ch.get(0.7,0)}/{ch.get(1.0,0)}/{ch.get(1.3,0)}  '
              f'留出: E→A {improved}, A→E {worsened}  (默认全数据 nA={nA0})', flush=True)
    out = pd.DataFrame(recs)
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print('结果已存:', OUT, flush=True)


if __name__ == '__main__':
    main()
