#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线结构健康监测系统 — 兼容入口（已拆分为 shm 包）

本文件为兼容入口，所有实现已拆分到 shm 包中：

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

保留本文件是为了兼容旧的导入方式（如 `from multi_source_shm import StreamProcessor`）。
推荐新代码直接使用 `from shm import ...`。

用法：
    python multi_source_shm.py dashboard --group 016 --speed 10 --port 5000
    python multi_source_shm.py batch --groups 016 017 018 019 020 --speed 10
"""
from shm import *  # noqa: F401,F403
from shm import __all__  # noqa: F401


if __name__ == '__main__':
    main()
