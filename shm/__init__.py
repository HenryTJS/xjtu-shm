#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shm 包：在线结构健康监测系统（多源流式处理）

将原 multi_source_shm.py（1455 行）拆分为模块化包：

    shm/
    ├── __init__.py          # 导出主接口 + 入口函数
    ├── config.py            # 常量、路径、全局配置
    ├── data_loader.py       # DataLoader
    ├── online_buffer.py     # OnlineBuffer
    ├── normalizer.py        # OnlineNormalizer
    ├── feature_extractor.py # OnlineFeatureExtractor
    ├── stage_divider.py     # OnlineStageDivider
    ├── anomaly_detector.py  # OnlineAnomalyDetector
    ├── streaming.py         # ChunkedDataReader + StreamSimulator + StreamProcessor
    ├── dashboard.py         # RealtimeDashboard
    └── templates/
        └── dashboard.html   # 前端 HTML 模板

用法：
    from shm import StreamProcessor, RealtimeDashboard
    from shm import run_online_dashboard, run_online_processing
"""
import argparse

from .config import (BASE_DIR, OUTPUT_DIR, TEMPLATE_DIR, DEFAULT_GROUPS,
                     NORMALIZER_WARMUP, NORMALIZER_WINDOW,
                     BASELINE_LENGTH, MIN_STABLE_POINTS, MIN_PHASE_DURATION,
                     COOLDOWN_DEFAULT, TRANSITION_THRESHOLDS,
                     STRAIN_JUMP_THRESHOLD, AE_SLOPE_RATIO_THRESHOLD, FO_DRIFT_THRESHOLD,
                     CHUNK_SIZE, DASHBOARD_DEFAULT_PORT, DASHBOARD_DEFAULT_SPEED)
from .data_loader import DataLoader
from .online_buffer import OnlineBuffer
from .normalizer import OnlineNormalizer
from .feature_extractor import OnlineFeatureExtractor
from .stage_divider import OnlineStageDivider
from .anomaly_detector import OnlineAnomalyDetector
from .streaming import ChunkedDataReader, StreamSimulator, StreamProcessor
from .dashboard import RealtimeDashboard

__all__ = [
    # 配置
    'BASE_DIR', 'OUTPUT_DIR', 'TEMPLATE_DIR', 'DEFAULT_GROUPS',
    'NORMALIZER_WARMUP', 'NORMALIZER_WINDOW',
    'BASELINE_LENGTH', 'MIN_STABLE_POINTS', 'MIN_PHASE_DURATION',
    'COOLDOWN_DEFAULT', 'TRANSITION_THRESHOLDS',
    'STRAIN_JUMP_THRESHOLD', 'AE_SLOPE_RATIO_THRESHOLD', 'FO_DRIFT_THRESHOLD',
    'CHUNK_SIZE', 'DASHBOARD_DEFAULT_PORT', 'DASHBOARD_DEFAULT_SPEED',
    # 组件
    'DataLoader', 'OnlineBuffer', 'OnlineNormalizer',
    'OnlineFeatureExtractor', 'OnlineStageDivider', 'OnlineAnomalyDetector',
    'ChunkedDataReader', 'StreamSimulator', 'StreamProcessor',
    'RealtimeDashboard',
    # 入口函数
    'run_online_dashboard', 'run_online_processing', 'main',
]


def run_online_dashboard(group_id='016', speed_factor=10.0, port=5000):
    """启动实时仪表盘"""
    dashboard = RealtimeDashboard(group_id, speed_factor, port)
    dashboard.run()


def run_online_processing(groups=None, speed_factor=10.0):
    """批量在线处理（无界面）"""
    if groups is None:
        groups = DEFAULT_GROUPS
    all_results = []
    for gid in groups:
        processor = StreamProcessor(gid, speed_factor=speed_factor)
        processor.load()
        results = processor.run_all()
        all_results.append(results)
        print(f'\n=== 组 {gid} 处理结果 ===')
        print(f'  总点数: {results["total_points"]}')
        print(f'  阶段分布: {results["stage_counts"]}')
        print(f'  异常点数: {results["anomaly_count"]}')
    return all_results


def main():
    parser = argparse.ArgumentParser(description='在线结构健康监测系统')
    parser.add_argument('mode', nargs='?', default='batch', choices=['dashboard', 'batch'],
                        help='运行模式: dashboard (实时仪表盘) / batch (批量处理)')
    parser.add_argument('--group', default='016', help='仪表盘模式: 组号 (默认 016)')
    parser.add_argument('--groups', nargs='+', default=DEFAULT_GROUPS,
                        help='批量模式: 组号列表')
    parser.add_argument('--speed', type=float, default=DASHBOARD_DEFAULT_SPEED,
                        help='模拟速度倍率 (默认 10)')
    parser.add_argument('--port', type=int, default=DASHBOARD_DEFAULT_PORT,
                        help='仪表盘端口 (默认 5000)')
    args = parser.parse_args()

    if args.mode == 'dashboard':
        run_online_dashboard(args.group, args.speed, args.port)
    else:
        run_online_processing(args.groups, args.speed)


if __name__ == '__main__':
    main()
