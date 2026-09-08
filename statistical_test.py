# -*- coding: utf-8 -*-
"""T9 统计显著性：主样本 6 组

- 方法对比：多源融合 D(full) vs 单源基线(仅 AE=no_strain, 仅应变=only_strain)，
  用同一 A-预警判据(不可逆 onset)产出每组预警点，与扩展参考 b2 对齐误差 |err| 配对比较。
- Wilcoxon signed-rank 配对检验（6 组小样本，注明局限）。
- Bootstrap(试件级重采样) 对 D 的 |err| 与断裂前提前量 lead 给 95% CI。
输出: results/statistical_test.csv + 控制台
"""
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy import stats

sys.path.insert(0, r'd:\lixiang')
os.chdir(r'd:\lixiang')
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex
from eval_common import GROUPS, per_group_metrics

CACHE = r'd:\lixiang\cache\_stats_cache'
OUT = r'd:\lixiang\results\statistical_test.csv'
MODES = {'full': '多源融合D(full)', 'no_strain': '仅AE(no_strain)',
         'only_strain': '仅应变(only_strain)'}
NBOOT = 2000


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


def wilcoxon_onesided(a, b):
    """H1: a < b(多源融合误差更小)。返回 p(单侧) 与 n 配对。"""
    diff = np.asarray(a, float) - np.asarray(b, float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n == 0:
        return np.nan, 0
    try:
        # 备择 alternative='less' => H1: a-b < 0 => a<b
        p = stats.wilcoxon(diff, alternative='less').pvalue
        return p, n
    except ValueError:
        # 全部相同(无法检验)视为不显著
        return np.nan, n


def bootstrap_ci(x, stat=np.mean, alpha=0.05):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    boot = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(NBOOT)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def main():
    modes = list(MODES.keys())
    jobs = [(g, m) for g in GROUPS for m in modes]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run_one, jobs))
    # 按 job 顺序重建 (gid, mode) -> 逐组指标
    per = {m: {} for m in modes}
    for (g, m), (gid, d) in zip(jobs, res):
        per[m][gid] = per_group_metrics(gid, d)
    rows = []
    for g in GROUPS:
        base = per['full'][g]
        row = {'gid': g, 'b2': base['b2'], 'b3': base['b3']}
        for m in modes:
            mm = per[m][g]
            row[f'{m}_t_warn'] = mm['t_warn']
            row[f'{m}_err'] = mm['err']
            row[f'{m}_lead'] = mm['lead']
            row[f'{m}_grade'] = mm['grade']
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8-sig')

    def colerr(m):
        return df[f'{m}_err'].to_numpy(float)

    print('=== 逐组 |预警-扩展| 误差(小=好) ===')
    print(df[['gid', 'full_err', 'no_strain_err', 'only_strain_err']]
          .assign(**{'|full|': df.full_err.abs(), '|AE|': df.no_strain_err.abs(),
                     '|strain|': df.only_strain_err.abs()})[['gid', '|full|', '|AE|', '|strain|']]
          .round(1).to_string(index=False))

    a_full = np.abs(colerr('full'))
    a_ae = np.abs(colerr('no_strain'))
    a_st = np.abs(colerr('only_strain'))
    print('\n=== Wilcoxon 配对检验(H1: 多源融合 |err| 更小) ===')
    p1, n1 = wilcoxon_onesided(a_full, a_ae)
    p2, n2 = wilcoxon_onesided(a_full, a_st)
    print(f'  D vs 仅AE   : p={p1:.3f} (n={n1})  均值|err| D={np.nanmean(a_full):.1f} vs AE={np.nanmean(a_ae):.1f}')
    print(f'  D vs 仅应变 : p={p2:.3f} (n={n2})  均值|err| D={np.nanmean(a_full):.1f} vs 应变={np.nanmean(a_st):.1f}')

    # 备注漏报(C: 无 onset)处理: 无 onset 的 err=None; 配对时排除, 并单独报告漏报数
    print('\n=== 漏报(C, 无不可逆onset)组数 ===')
    for m in modes:
        print(f'  {MODES[m]}: {int((df[f"{m}_grade"]=="C").sum())}')

    print('\n=== Bootstrap 95% CI(主样本 6 组, 试件重采样) ===')
    lo, hi = bootstrap_ci(a_full)
    print(f'  D  |err| 均值 CI: [{lo:.1f}, {hi:.1f}]  样本均值 {np.nanmean(a_full):.1f}')
    lead = df['full_lead'].to_numpy(float)
    lo2, hi2 = bootstrap_ci(lead)
    print(f'  D  断裂前提前量 lead 均值 CI: [{lo2:.1f}, {hi2:.1f}]  样本均值 {np.nanmean(lead):.1f}')
    print('  (小样本 n=6，CI 仅示意；结论需谨慎解读)')
    print('结果已存:', OUT)


if __name__ == '__main__':
    main()
