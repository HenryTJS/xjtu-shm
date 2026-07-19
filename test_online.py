#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速测试：验证改造后的在线流式处理系统"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置 stdout 编码为 utf-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import argparse
from multi_source_shm import StreamProcessor
import numpy as np

# 命令行参数
parser = argparse.ArgumentParser(description='测试在线流式处理系统')
parser.add_argument('--group', default='016', help='组号 (默认 016，支持 001-027)')
parser.add_argument('--max-points', type=int, default=1500, help='最大处理点数 (默认 1500)')
parser.add_argument('--speed', type=float, default=10000, help='模拟速度倍率 (默认 10000)')
args = parser.parse_args()

print('=' * 60)
print(f'测试改造后的在线流式处理系统 - 组 {args.group}')
print('=' * 60)

# 初始化处理器
proc = StreamProcessor(args.group, speed_factor=args.speed)
n = proc.load()
print(f'\n加载完成: {n} 点')

# 处理指定数量的点
count = 0
while proc.simulator.has_next() and count < args.max_points:
    result = proc.process_step()
    if result is None:
        break
    count += 1
    if count in [1, 50, 100, 150, 200, 250, 300, 500, 800, 1000, 1200, 1500]:
        ae_raw_str = f'{result["ae_energy_raw"]:.0f}' if result["ae_energy_raw"] is not None else 'N/A'
        fo_raw_str = f'{result["fo_mean_raw"]:.2f}' if result["fo_mean_raw"] is not None else 'N/A'
        spike_str = f'{result["ae_spike_rate"]:.4f}' if result["ae_spike_rate"] is not None else 'N/A'
        print(f'  点{result["index"]:>5d}: 阶段={result["stage"]}, '
              f'应变={result["strain"]:.4f}, 异常={result["anomaly"]}, '
              f'AE原始能量={ae_raw_str}, FO原始均值={fo_raw_str}, '
              f'AE尖峰率={spike_str}')

# 直接从 stages 列表计算阶段分布
stages_arr = np.array(proc.results['stages'])
stage_counts = {
    '0': int(np.sum(stages_arr == 0)),
    '1': int(np.sum(stages_arr == 1)),
    '2': int(np.sum(stages_arr == 2)),
    '3': int(np.sum(stages_arr == 3)),
}
anomaly_count = int(np.sum(proc.results['anomalies']))

print(f'\n处理完成: {count} 点')
print(f'阶段分布: {stage_counts}')
print(f'异常点数: {anomaly_count}')
proc.simulator.cleanup()
print('临时文件已清理')
print('\n测试完成!')
