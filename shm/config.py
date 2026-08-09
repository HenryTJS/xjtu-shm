#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局配置：路径、常量、默认参数

将 multi_source_shm.py 中的路径与硬编码参数集中管理，
便于统一调整阈值、窗口大小等参数。
"""
import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 默认组号
# ============================================================
DEFAULT_GROUPS = ['016', '017', '018', '019', '020']

# ============================================================
# 在线归一化参数
# ============================================================
NORMALIZER_WARMUP = 100          # 归一化 warm-up 点数
NORMALIZER_WINDOW = 200          # 归一化滑动窗口大小

# ============================================================
# 阶段划分参数（OnlineStageDivider）
# ============================================================
BASELINE_LENGTH = 500            # 基线收集长度
MIN_STABLE_POINTS = 500          # 前 N 点不允许任何跃迁
MIN_PHASE_DURATION = 300         # 每个阶段至少维持点数
COOLDOWN_DEFAULT = 500           # 默认冷却周期
# 默认跃迁阈值（Phase 2 阈值降低以允许 AE 尖峰触发 Phase 3）
TRANSITION_THRESHOLDS = {0: 0.60, 1: 0.65, 2: 0.60}
# 默认检测阈值
STRAIN_JUMP_THRESHOLD = 0.5
AE_SLOPE_RATIO_THRESHOLD = 3.0
FO_DRIFT_THRESHOLD = 0.5

# ============================================================
# 流式读取参数
# ============================================================
CHUNK_SIZE = 1000                # 分块读取大小

# ============================================================
# 仪表盘参数
# ============================================================
DASHBOARD_DEFAULT_PORT = 5000
DASHBOARD_DEFAULT_SPEED = 10.0
