# -*- coding: utf-8 -*-
"""在线 vs 离线 阶段划分对比

支持三种运行模式：
  online    纯在线（逐点流式，含实时指示器/预警样例）
  offline   纯离线（批处理基准，应变水平变点）
  compare   在线离线同步运行并比较二者（默认）

用法：
  python compare.py --mode online  --groups 016 018
  python compare.py --mode offline --groups 016 018
  python compare.py --mode compare --groups 016 018        # 默认
  python compare.py --mode compare --all                   # 全部有效组(006-014,016-027)
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import numpy as np
from shm import StreamProcessor
from shm.data_loader import DataLoader
from shm.batch_divider import BatchStageDivider

# 有效试件：排除 001-005、015（可能存在问题的组）
ALL_GROUPS = [f'{i:03d}' for i in range(6, 28) if i != 15]
DEFAULT_GROUPS = ['006', '010', '016', '018', '020', '023']


def first_transition(stages, p):
    idx = np.where(stages == p)[0]
    return int(idx[0]) if len(idx) else None


def stage_pct(stages):
    n = len(stages)
    if n == 0:
        return {str(i): 0.0 for i in range(4)}
    return {str(i): round(100 * float(np.sum(stages == i)) / n, 1) for i in range(4)}


def run_online(gid):
    """在线逐点流式：返回 (阶段序列, 异常序列, CPU耗时, 最后一个结果dict)"""
    proc = StreamProcessor(gid, speed_factor=1e6)
    proc.load()
    online_cpu = 0.0
    last_result = None
    while proc.simulator.has_next():
        t1 = time.time()
        r = proc.process_step()
        online_cpu += time.time() - t1
        if r is not None:
            last_result = r
    stages = np.array(proc.results['stages'])
    anomalies = np.array(proc.results['anomalies'])
    proc.simulator.cleanup()
    return stages, anomalies, online_cpu, last_result


def run_offline(gid):
    """离线批处理：返回 (阶段序列, 变点索引, 耗时)"""
    dl = DataLoader(gid).load_all()
    data = dl.sync_timeline()
    strain = data['strain'].values
    bd = BatchStageDivider()
    t0 = time.time()
    stages, pts = bd.divide(strain)
    elapsed = time.time() - t0
    return stages, pts, elapsed


def mode_online(gid):
    stages, anomalies, cpu, last = run_online(gid)
    n = len(stages)
    trans = [first_transition(stages, p) for p in (1, 2, 3)]
    trans_s = [round(100 * t / n, 1) if t is not None else None for t in trans]
    ind = ''
    if last is not None:
        ind = (f" | 末点: 阶段={last['phase_name']} 预警级别={last['alert_level']} "
               f"连续异常={last['anomaly_run']} 异常率={last['anomaly_rate']*100:.1f}%")
    print(f"[在线] {gid}: n={n} 阶段分布={stage_pct(stages)} "
          f"跃迁@%={trans_s} 异常={int(anomalies.sum())} 耗时={cpu:.2f}s{ind}")


def mode_offline(gid):
    stages, pts, elapsed = run_offline(gid)
    n = len(stages)
    pts_s = [round(100 * p / n, 1) for p in pts]
    print(f"[离线] {gid}: n={n} 阶段分布={stage_pct(stages)} "
          f"变点@%={pts_s} 耗时={elapsed*1000:.1f}ms")


def mode_compare(gid):
    """在线离线同步运行并比较二者"""
    on_stages, _, on_cpu, _ = run_online(gid)
    off_stages, off_pts, off_time = run_offline(gid)
    n = min(len(on_stages), len(off_stages))
    on_s, off_s = on_stages[:n], off_stages[:n]
    consistency = float(np.mean(on_s == off_s))
    on_t = [first_transition(on_s, p) for p in (1, 2, 3)]
    off_t = [first_transition(off_s, p) for p in (1, 2, 3)]
    on_s_ = [round(100 * t / n, 1) if t is not None else None for t in on_t]
    off_s_ = [round(100 * t / n, 1) if t is not None else None for t in off_t]
    # 跃迁延迟（在线相对离线，%寿命差，正值=在线偏早）
    delay = None
    for a, b in zip(on_t, off_t):
        if a is not None and b is not None:
            delay = round(100 * (b - a) / n, 1)  # 取第一个都有定义的跃迁
            break
    print(f"[对比] {gid}: n={n} 一致性={100*consistency:.1f}% "
          f"在线耗时={on_cpu:.2f}s 离线耗时={off_time*1000:.0f}ms "
          f"在线跃迁@%={on_s_} 离线跃迁@%={off_s_} 在线相对离线延迟@%={delay}")


def main():
    parser = argparse.ArgumentParser(description='在线 vs 离线 阶段划分')
    parser.add_argument('--mode', choices=['online', 'offline', 'compare'], default='compare',
                        help='运行模式: online(纯在线) / offline(纯离线) / compare(在线离线比较,默认)')
    parser.add_argument('--groups', nargs='+', default=None, help='试件号列表')
    parser.add_argument('--all', action='store_true', help='运行全部有效组(006-014,016-027)')
    args = parser.parse_args()

    groups = args.groups if args.groups else (ALL_GROUPS if args.all else DEFAULT_GROUPS)

    mode_map = {'online': mode_online, 'offline': mode_offline, 'compare': mode_compare}
    print(f'===== 模式: {args.mode} | 试件: {groups} =====')
    for gid in groups:
        try:
            mode_map[args.mode](gid)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{gid}: 失败 - {e}")


if __name__ == '__main__':
    main()
