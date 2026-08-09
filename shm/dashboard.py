#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时仪表盘：Flask SSE 推送 + Chart.js 前端渲染

前端 HTML 模板从 shm/templates/dashboard.html 读取，
后端通过 Server-Sent Events (SSE) 逐点推送处理结果。
"""
import json
import os
import queue
import threading

from flask import Flask, Response, jsonify

from .config import TEMPLATE_DIR
from .streaming import StreamProcessor


def _load_html_template():
    """从 templates/dashboard.html 读取前端模板"""
    template_path = os.path.join(TEMPLATE_DIR, 'dashboard.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


class RealtimeDashboard:
    """实时仪表盘：Flask SSE 推送 + Chart.js 前端渲染"""

    def __init__(self, group_id, speed_factor=10.0, port=5000):
        self.group_id = group_id
        self.speed_factor = speed_factor
        self.port = port
        self.processor = StreamProcessor(group_id, speed_factor)
        self.data_queue = queue.Queue()
        self._running = False
        self._html_template = _load_html_template()

    def _make_sse_app(self):
        app = Flask(__name__)

        @app.route('/')
        def index():
            return self._html_template.replace('{{GROUP_ID}}', self.group_id)

        @app.route('/stream')
        def stream():
            def generate():
                yield f'data: {json.dumps({"type": "meta", "group_id": self.group_id, "speed": self.speed_factor})}\n\n'
                if not self._running:
                    self._running = True
                    t = threading.Thread(target=self._run_processing, daemon=True)
                    t.start()
                while True:
                    try:
                        data = self.data_queue.get(timeout=1)
                        yield f'data: {json.dumps(data)}\n\n'
                    except queue.Empty:
                        yield ': keepalive\n\n'
            return Response(generate(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'Access-Control-Allow-Origin': '*'})

        @app.route('/status')
        def status():
            return jsonify({'running': self._running, 'progress': self.processor.simulator.progress})

        return app

    def _run_processing(self):
        self.processor.load()
        self.processor.run_all(callback=self._on_data)

    def _on_data(self, result):
        self.data_queue.put(result)

    def run(self):
        print(f'\n>>> 启动实时仪表盘: 组 {self.group_id} @ http://localhost:{self.port}')
        print(f'    模拟速度: {self.speed_factor}x')
        app = self._make_sse_app()
        app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
