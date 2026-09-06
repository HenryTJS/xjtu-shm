#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局配置（精简）：连续损伤度 D(t) 体系的必要常量

四阶段划分(stage_divider)体系已弃用(2026-09-06)，相关配置一并移除。
保留被 shm 模块实际引用的：BASE_DIR / CHUNK_SIZE / EXT_BLOCK_PTS / DEFAULT_GROUPS。
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 主样本组（6 组，正式口径）
DEFAULT_GROUPS = ['016', '017', '018', '019', '020', '022']

CHUNK_SIZE = 1000            # 逐块数据读取大小(点)

EXT_BLOCK_PTS = 500          # 损伤度 D 分块点数（≈每块结算一次块级统计）
