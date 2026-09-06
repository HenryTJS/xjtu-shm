# -*- coding: utf-8 -*-
"""T7 参数敏感性(±30%)——17 组全量

对 D(t) 关键参数(每个独立扰动 ±30%, 其余默认)重算 17 组,
报告聚合指标相对默认的变化 → 判断"参数不敏感"或指出敏感点。

运行: python parameter_sensitivity.py [--workers N]
输出: results/parameter_sensitivity.csv
缓存: cache/_t7_cache/<cfg>/  (按参数指纹隔离, 可续跑)
"""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (PARS, DEFAULT, cfg_id, run_cfg, table_for,
                         summary_row, GROUPS)

OUT = os.path.join(ROOT_DIR := r'd:\lixiang\results', 'parameter_sensitivity.csv')


def build_configs():
    """默认 + 每参数 ±30%。返回 [(cfg_params, label)]"""
    cfgs = [({}, 'default')]
    for p in PARS:
        for m in (0.7, 1.3):
            cp = dict(DEFAULT)
            cp[p] = DEFAULT[p] * m
            cfgs.append((cp, f'{p}x{m}'))
    return cfgs


def main():
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
    cfgs = build_configs()
    rows = []
    print(f'[T7-sens] 共 {len(cfgs)} 配置 × {len(GROUPS)} 组, workers={workers}', flush=True)
    t0 = time.time()
    for params, label in cfgs:
        cid = cfg_id(params)
        ts = time.time()
        gid_d = run_cfg(params, workers=workers)
        df = table_for(cid, gid_d)
        row = summary_row(cid, df)
        row['label'] = label
        rows.append(row)
        print(f'  [{label}] {time.time()-ts:.0f}s  n55={row["n55"]} n85={row["n85"]} '
              f'A={row["nA"]} E={row["nE"]} D={row["nD"]} C={row["nC"]} '
              f'med|err|={row["mean_abs_err"] if row["mean_abs_err"]==row["mean_abs_err"] else "-"}', flush=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'总耗时 {time.time()-t0:.0f}s; 结果已存: {OUT}', flush=True)


if __name__ == '__main__':
    main()
