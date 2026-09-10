# -*- coding: utf-8 -*-
"""导出疲劳机多源在线监测看板数据
=====================================

把 6 组主样本(016/017/018/019/020/022)的**逐点在线流式**结果
(连续损伤度 D(t)、证据层 e_ae/e_strain/risk、分级 level、各源原始信号)
降采样打包为前端直接可用的 JS 数据文件。

严格复用正式口径(见 README):
  - 流式入口 : shm.streaming.StreamSimulator     (逐点, 零未来信息)
  - 损伤模型 : shm.damage_index.OnlineDamageIndex (默认参数 = 正式方法)

输出:
  dashboard/data/{gid}.js   每组一个数据包 (window.SHM_DATA[gid] = {...})
  dashboard/data/index.js   组清单与摘要 (window.SHM_INDEX = [...])

数值编码(压缩体积, 前端解码):
  D / risk / e_ae / e_strain : ×1000 整数
  strain / fo                : ×100 整数
  aeLog                      : log10(peak)×1000 整数, 无事件 = -99999
  aeN                        : 帧内 AE 事件数
  level                      : 0/1/2/3 整数

作者: Roo   日期: 2026-09-10
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))   # 项目根(含 shm)
sys.path.insert(0, ROOT)                     # main/(eval_common)
os.chdir(ROOT)

from eval_common import ref_map                      # noqa: E402
from shm.config import DEFAULT_GROUPS                # noqa: E402
from shm.damage_index import OnlineDamageIndex       # noqa: E402
from shm.streaming import StreamSimulator            # noqa: E402

OUTDIR = os.path.join(ROOT, 'dashboard', 'data')
STEP = 5                 # 降采样步长(点): 10Hz → 2Hz 帧
AE_EMPTY = -99999        # aeLog 空值标记
LEVEL_NAMES = ['正常', '注意', '预警', '临危']


def _i100(x):
    return int(round(float(x) * 100.0))


def _i1000(x):
    return int(round(float(x) * 1000.0))


def _pack(gid):
    """流式跑一组 → 数据包(dict)。"""
    sim = StreamSimulator(gid)
    n = sim.load_data()
    di = OnlineDamageIndex()
    nfr = (n + STEP - 1) // STEP

    T = np.zeros(nfr, dtype=np.float64)
    Df = np.zeros(nfr, dtype=np.int32)
    RISK = np.zeros(nfr, dtype=np.int32)
    EAE = np.zeros(nfr, dtype=np.int32)
    EST = np.zeros(nfr, dtype=np.int32)
    LV = np.zeros(nfr, dtype=np.int16)
    ST = np.zeros(nfr, dtype=np.int32)
    AEL = np.full(nfr, AE_EMPTY, dtype=np.int32)
    AEN = np.zeros(nfr, dtype=np.int32)
    FO = {c: np.zeros(nfr, dtype=np.int32) for c in sim.fo_cols}
    Dfull = np.zeros(n, dtype=np.float32)

    i = 0
    while sim.has_next():
        p = sim.next_point()
        f = i // STEP
        peak = None
        if p.get('ae_new') and p.get('ae'):
            try:
                peak = float(p['ae'].get('ae_Peak', 0.0) or 0.0)
            except (TypeError, ValueError):
                peak = 0.0
        d = di.update(p['strain'], peak)
        Dfull[i] = d

        T[f] = float(p['time'])
        Df[f] = _i1000(d)
        RISK[f] = _i1000(di.risk)
        EAE[f] = _i1000(di._last_e_ae)
        EST[f] = _i1000(di._last_e_strain)
        LV[f] = int(di.level)
        if p['strain'] is not None:
            ST[f] = _i100(p['strain'])
        fo = p.get('fo') or {}
        for c, arr in FO.items():
            v = fo.get(c)
            if v is not None:
                arr[f] = _i100(v)
        if peak is not None and peak > 0:
            lg = _i1000(np.log10(peak))
            AEL[f] = lg if AEL[f] == AE_EMPTY else max(AEL[f], lg)
            AEN[f] += 1
        i += 1

    sim.cleanup()

    # --- 摘要统计(全分辨率 D) ---
    def first_ge(th):
        idx = np.where(Dfull >= th)[0]
        return round(float(idx[0]) / n * 100.0, 1) if len(idx) else None

    refs = ref_map().get(gid, (None, 99.0))
    t25, t55, t85 = first_ge(0.25), first_ge(0.55), first_ge(0.85)
    dur = float(T[nfr - 1])

    # --- 预警事件(level 首次升级点) ---
    warn = []
    last = 0
    for k in range(nfr):
        lv = int(LV[k])
        if lv > last:
            for g in range(last + 1, lv + 1):
                warn.append({'f': k, 't': round(float(T[k]), 1), 'lv': g})
            last = lv

    pkg = {
        'gid': gid,
        'n': int(n),
        'nfr': int(nfr),
        'step': STEP,
        'dt': round(dur / max(nfr - 1, 1), 4),
        'dur': round(dur, 1),
        'dtReal': round(dur / max(n - 1, 1), 4),
        'foCols': list(FO.keys()),
        'meta': {
            'D_end': round(float(Dfull[-1]), 3),
            't25': t25, 't55': t55, 't85': t85,
            'b2': refs[0] if refs[0] is not None else None,
            'b3': refs[1] if refs[1] is not None else None,
            'aeEvents': int(AEN.sum()),
            'nFo': len(FO),
        },
        'warn': warn,
        # --- 波形数组 ---
        'D': Df.tolist(),
        'risk': RISK.tolist(),
        'eae': EAE.tolist(),
        'est': EST.tolist(),
        'lv': LV.tolist(),
        'st': ST.tolist(),
        'ael': AEL.tolist(),
        'aen': AEN.tolist(),
        't': [round(float(x), 1) for x in T.tolist()],
        'fo': {c: a.tolist() for c, a in FO.items()},
    }
    return pkg


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    groups = sys.argv[1:] or list(DEFAULT_GROUPS)
    index = []
    for gid in groups:
        print(f'\n=== 导出 {gid} ===')
        pkg = _pack(gid)
        path = os.path.join(OUTDIR, f'{gid}.js')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('window.SHM_DATA=window.SHM_DATA||{};')
            f.write(f'window.SHM_DATA["{gid}"]=')
            f.write(json.dumps(pkg, ensure_ascii=False, separators=(',', ':')))
            f.write(';\n')
        size = os.path.getsize(path) / 1024.0
        print(f'  写出 {path}  ({size:.0f} KB)')
        print(f'  帧数={pkg["nfr"]} 时长={pkg["dur"]}s D_end={pkg["meta"]["D_end"]} '
              f't25={pkg["meta"]["t25"]} t55={pkg["meta"]["t55"]} t85={pkg["meta"]["t85"]} '
              f'AE事件={pkg["meta"]["aeEvents"]} 光纤通道={pkg["meta"]["nFo"]}')
        index.append({'gid': gid, 'nfr': pkg['nfr'], 'dur': pkg['dur'],
                      'n': pkg['n'], 'meta': pkg['meta'], 'foCols': pkg['foCols'],
                      'warn': pkg['warn']})
    ipath = os.path.join(OUTDIR, 'index.js')
    with open(ipath, 'w', encoding='utf-8') as f:
        f.write('window.SHM_INDEX=')
        f.write(json.dumps(index, ensure_ascii=False, separators=(',', ':')))
        f.write(';\n')
    print(f'\n写出清单 {ipath}')


if __name__ == '__main__':
    main()
