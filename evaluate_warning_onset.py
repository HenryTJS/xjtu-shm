# -*- coding: utf-8 -*-
"""A: "不可逆上升"预警 onset（不依赖绝对高阈 0.85，改用低阈 0.3 + 不回落实 确认）

思路: 健康期 D 处于低位平台; 损伤/扩展使 D 净上升离开平台。onset = D 首次≥0.3
且此后 hold 窗内未回落到 <0.3-0.15(=0.15) 的"不可逆"点——早期加载/磨合若造成
D 短暂升高但随后回落(健康期长), 此 onset 作废, 等到真正不可逆恶化才预警。

评估: t_warn vs 弱标签 b2(err)/b3(lead), 分级(不过早不过晚)。
输出: results/warning_onset.csv（读取 cache/_hi_cache，先跑 evaluate_damage_degree）
"""
import os, sys
sys.path.insert(0, r'd:\lixiang')
os.chdir(r'd:\lixiang')
import numpy as np
import pandas as pd

GROUPS = ['016', '017', '018', '019', '020', '022']   # 主样本 6 组
DNCACHE = r'd:\lixiang\cache\_hi_cache'
OUT = r'd:\lixiang\results\warning_onset.csv'
LOW = 0.30          # 预警触发低阈
DROP = 0.15         # 允许回落幅度(低于 LOW-DROP 视为回到平台, onset 作废)
HOLD_FRAC = 0.02    # 确认窗 = 寿命的 2%(点数)


def onset_of(d, hold):
    """返回 onset 索引(life%), 或 None。d: 逐点 D"""
    n = len(d)
    low, drop = LOW, LOW - DROP
    i = 0
    while i < n:
        if d[i] >= low:
            j_end = min(n - 1, i + hold)
            seg = d[i:j_end + 1]
            if seg.min() >= drop:
                return float(i) / n * 100.0
            # 确认失败: 跳过已检查段, 从回落到 <drop 之后继续找
            below = np.where(seg < drop)[0]
            i = i + (below[0] if len(below) else hold)
        else:
            i += 1
    return None


def load_ref(gid):
    sm = pd.read_csv(r'd:\lixiang\weak_labels\labels_summary.csv', encoding='utf-8-sig')
    sm['gid'] = sm['gid'].astype(int).map(lambda x: f'{x:03d}')
    r = sm[sm['gid'] == gid].iloc[0]
    return float(r['b2']), float(r['b3'])


def main():
    rows = []
    for gid in GROUPS:
        d = np.load(os.path.join(DNCACHE, f'{gid}.npy'))
        n = len(d)
        hold = max(2000, int(n * HOLD_FRAC))
        tw = onset_of(d, hold)
        b2, b3 = load_ref(gid)
        err = round(tw - b2, 1) if tw is not None else None
        lead = round(b3 - tw, 1) if tw is not None else None
        if tw is None:
            grade = 'C漏报'
        elif err is not None and err < -15:
            grade = 'E过早'
        elif err is not None and err > 15:
            grade = 'D偏晚'
        else:
            grade = 'A合理'
        rows.append(dict(gid=gid, t_warn=tw, b2=b2, b3=b3,
                         err=err, lead=lead, grade=grade))
        print(f'{gid}: t_warn={tw}  err={err}  lead={lead}  [{grade}]', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    cnt = df['grade'].value_counts()
    print('=== 分级计数 ===')
    for k in ['A合理', 'D偏晚', 'E过早', 'C漏报']:
        print(f'  {k}: {int(cnt.get(k, 0))} 组')
    ok = df[df['grade'] == 'A合理']
    if len(ok):
        print(f'A 组平均 lead(断裂前提前)={ok["lead"].mean():.1f}%')
    print('结果已存:', OUT)


if __name__ == '__main__':
    main()
