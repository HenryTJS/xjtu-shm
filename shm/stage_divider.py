#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线阶段划分状态机：逐点决策，单调跃迁 0→1→2→3

基于数据驱动策略：
- 应变: 滑动均值 LEVEL 变化 + 原始值阈值
- AE:   累积能量加速比（短窗/长窗斜率比）+ 事件率 + 异常分数 + 原始能量尖峰
- FO:   均值偏移比例

【改造3】自适应阈值：基于基线（warm-up）统计量动态计算阈值
【修复5】强制最小稳定期 + 每阶段最小持续点数 + 冷却周期
【修复6】Phase 2→3 触发：降低 AE event_rate 上限、降低 Phase 2 阈值、提高 spike 检测灵敏度
"""
import numpy as np

from .online_buffer import OnlineBuffer
from .config import (BASELINE_LENGTH, MIN_STABLE_POINTS, MIN_PHASE_DURATION,
                     COOLDOWN_DEFAULT, TRANSITION_THRESHOLDS,
                     STRAIN_JUMP_THRESHOLD, AE_SLOPE_RATIO_THRESHOLD, FO_DRIFT_THRESHOLD)


class OnlineStageDivider:
    """在线阶段划分状态机：逐点决策，单调跃迁 0→1→2→3
    基于数据驱动策略：
    - 应变: 滑动均值 LEVEL 变化 + 原始值阈值
    - AE:   累积能量加速比（短窗/长窗斜率比）
    - FO:   均值偏移比例

    【改造3】自适应阈值：基于基线（warm-up）统计量动态计算阈值
    """

    def __init__(self):
        self.current_phase = 0
        self.phase_history = []
        # 应变
        self.strain_buffer = OnlineBuffer([50, 100, 200, 500])
        self.strain_baseline_mean = None
        self.strain_warm_count = 0
        # AE
        self.ae_energy_buffer = OnlineBuffer([50, 100, 200, 500])
        self.energy_accum = 0.0
        self.energy_max = 0.0
        self.prev_energy = 0.0
        self.energy_slopes = []
        # FO
        self.fo_mean_buffer = OnlineBuffer([50, 100, 200, 500])
        self.fo_baseline_mean = None
        self.fo_warm_count = 0
        # 跃迁控制
        self.transition_cooldown = 0
        # ========== 改造3: 自适应阈值 ==========
        # 基线统计量（warm-up 阶段收集）
        self.strain_baseline_values = []      # 应变基线值列表
        self.strain_diff_baseline = []        # 应变差分基线
        self.fo_baseline_values = []          # FO 基线值列表
        self.ae_slope_baseline = []           # AE 斜率基线
        self.baseline_collected = False       # 基线是否收集完成
        self.baseline_length = BASELINE_LENGTH  # 基线长度（从200增加到500，确保基线稳定）
        # 自适应阈值（由基线统计量计算得出）
        self.strain_jump_threshold = STRAIN_JUMP_THRESHOLD  # 默认值（从0.3提高到0.5，防止早期误触发）
        self.ae_slope_ratio_threshold = AE_SLOPE_RATIO_THRESHOLD  # 默认值（从2.0提高到3.0）
        self.fo_drift_threshold = FO_DRIFT_THRESHOLD  # 默认值（从0.25提高到0.5）
        self.cooldown_period = COOLDOWN_DEFAULT  # 默认值（从200提高到500）
        self.transition_thresholds = dict(TRANSITION_THRESHOLDS)  # Phase 2 默认阈值降低到 0.60
        # 基线变化率（用于自适应 cooldown）
        self.baseline_strain_std = None
        self.baseline_fo_std = None
        # ========== 修复5: 强制最小稳定期 ==========
        self.min_stable_points = MIN_STABLE_POINTS  # 前500点不允许任何跃迁
        self.phase_entry_points = {0: 0}  # 记录每个阶段的进入点
        # 调试
        self.total_points = 0
        self.debug_info = []

    def _update_buffers(self, strain_val, features):
        if strain_val is not None and not np.isnan(strain_val):
            self.strain_buffer.push(strain_val)
        if 'ae_cumulative_energy' in features:
            self.energy_accum = features['ae_cumulative_energy']
            self.ae_energy_buffer.push(self.energy_accum)
            if self.energy_accum > self.energy_max:
                self.energy_max = self.energy_accum
        if 'fo_mean' in features:
            self.fo_mean_buffer.push(features['fo_mean'])

    # ========== 改造3: 自适应基线收集 ==========
    def _collect_baseline(self, strain_val, features):
        """收集 warm-up 阶段的基线数据"""
        if self.baseline_collected:
            return

        # 收集应变基线
        if strain_val is not None and not np.isnan(strain_val):
            self.strain_baseline_values.append(strain_val)
            if len(self.strain_baseline_values) >= 2:
                self.strain_diff_baseline.append(
                    abs(strain_val - self.strain_baseline_values[-2]))

        # 收集 FO 基线
        if 'fo_mean' in features:
            self.fo_baseline_values.append(features['fo_mean'])

        # 收集 AE 斜率基线
        if 'ae_cumulative_energy' in features:
            current_energy = features['ae_cumulative_energy']
            slope = current_energy - self.prev_energy if self.prev_energy > 0 else 0.0
            self.prev_energy = current_energy
            if slope > 0:
                self.ae_slope_baseline.append(slope)

        # 判断基线是否收集完成
        if (len(self.strain_baseline_values) >= self.baseline_length and
                len(self.fo_baseline_values) >= self.baseline_length // 2):
            self._compute_adaptive_thresholds()
            self.baseline_collected = True
            print(f'    [自适应阈值] 基线收集完成, 计算阈值:')
            print(f'      应变跳变阈值={self.strain_jump_threshold:.3f}, '
                  f'AE斜率比阈值={self.ae_slope_ratio_threshold:.2f}, '
                  f'FO漂移阈值={self.fo_drift_threshold:.3f}')
            print(f'      冷却周期={self.cooldown_period}, '
                  f'跃迁阈值={self.transition_thresholds}')

    def _compute_adaptive_thresholds(self):
        """基于基线统计量计算自适应阈值"""
        # --- 应变跳变阈值 ---
        strain_arr = np.array(self.strain_baseline_values)
        strain_std = float(np.std(strain_arr))
        strain_mean = float(np.mean(strain_arr))
        self.baseline_strain_std = strain_std

        # ========== 修复5: 更保守的应变跳变阈值 ==========
        if strain_mean > 1e-6:
            cv = strain_std / strain_mean
            self.strain_jump_threshold = max(5.0 * cv, 0.3)
        else:
            self.strain_jump_threshold = 0.5

        # 应变差分基线
        if len(self.strain_diff_baseline) > 10:
            diff_arr = np.array(self.strain_diff_baseline)
            diff_p95 = float(np.percentile(diff_arr, 95))
            diff_mean = float(np.mean(diff_arr))
            self.strain_jump_threshold = max(
                self.strain_jump_threshold,
                8.0 * diff_mean if diff_mean > 1e-10 else 0.5,
                3.0 * diff_p95 if diff_p95 > 1e-10 else 0.5
            )
        self.strain_jump_threshold = max(0.3, min(1.5, self.strain_jump_threshold))

        # --- AE 斜率比阈值 ---
        if len(self.ae_slope_baseline) > 20:
            slope_arr = np.array(self.ae_slope_baseline)
            slope_mean = float(np.mean(slope_arr))
            slope_std = float(np.std(slope_arr))
            if slope_mean > 1e-12:
                self.ae_slope_ratio_threshold = max(
                    (slope_mean + 5.0 * slope_std) / slope_mean,
                    2.0
                )
            else:
                self.ae_slope_ratio_threshold = 3.0
        else:
            self.ae_slope_ratio_threshold = 3.0
        self.ae_slope_ratio_threshold = max(1.5, min(8.0, self.ae_slope_ratio_threshold))

        # --- FO 漂移阈值 ---
        if len(self.fo_baseline_values) > 10:
            fo_arr = np.array(self.fo_baseline_values)
            fo_std = float(np.std(fo_arr))
            fo_mean = float(np.mean(fo_arr))
            self.baseline_fo_std = fo_std
            if fo_mean > 1e-6:
                fo_cv = fo_std / fo_mean
                self.fo_drift_threshold = max(6.0 * fo_cv, 0.3)
            else:
                self.fo_drift_threshold = 0.5
        else:
            self.fo_drift_threshold = 0.5
        # FO 阈值上限从 1.5 降低到 1.0，因为 Z-score 标准化后漂移通常不超过 1.0 标准差
        self.fo_drift_threshold = max(0.3, min(1.0, self.fo_drift_threshold))

        # --- 自适应冷却周期 ---
        if self.baseline_strain_std is not None:
            base_cv = self.baseline_strain_std / max(abs(strain_mean), 1e-6)
            self.cooldown_period = int(max(300, min(800, 300 + base_cv * 2000)))
        else:
            self.cooldown_period = 500

        # --- 自适应跃迁阈值（更保守，但 Phase 2 阈值降低以允许 AE 尖峰触发） ---
        base_noise_level = self.baseline_strain_std / max(abs(strain_mean), 1e-6) if strain_mean != 0 else 0.1
        noise_factor = 1.0 + min(base_noise_level, 1.0)
        self.transition_thresholds = {
            0: max(0.40, min(0.70, 0.50 * noise_factor)),
            1: max(0.45, min(0.75, 0.55 * noise_factor)),
            2: max(0.40, min(0.70, 0.50 * noise_factor)),  # Phase 2 阈值降低，允许 AE 尖峰触发 Phase 3
        }

    def _detect_strain_jump(self, strain_val, features):
        """检测应变 LEVEL 变化（使用自适应阈值 + 原始应变值）"""
        if strain_val is None or np.isnan(strain_val):
            return 0.0
        self.strain_warm_count += 1
        # ========== 修复5: 延长 warm-up 到 200 点 ==========
        if self.strain_warm_count <= 200:
            if self.strain_baseline_mean is None:
                self.strain_baseline_mean = strain_val
            else:
                self.strain_baseline_mean = 0.995 * self.strain_baseline_mean + 0.005 * strain_val
            return 0.0
        # 方法1: 滑动均值相对基线偏移（使用归一化值）
        if self.strain_buffer.is_warm(50):
            m, _ = self.strain_buffer.get_mean_std(50)
            if m is not None:
                denom = max(abs(self.strain_baseline_mean), 1e-6)
                level_shift = abs(m - self.strain_baseline_mean) / denom
                if level_shift > self.strain_jump_threshold:
                    return min((level_shift - self.strain_jump_threshold) / 2.0, 1.0)
        # 方法2: 使用原始应变值（非归一化值）检测大幅跳变
        raw_original = features.get('strain_raw_original', None)
        if raw_original is not None and not np.isnan(raw_original):
            denom_raw = max(abs(self.strain_baseline_mean), 1e-6)
            raw_shift = abs(raw_original - self.strain_baseline_mean) / denom_raw
            if raw_shift > self.strain_jump_threshold * 3.0:
                return min((raw_shift - self.strain_jump_threshold * 3.0) / 3.0, 1.0)
        return 0.0

    def _detect_ae_transition(self, features):
        """检测AE阶段跃迁：多指标融合（能量加速比 + 事件率突增 + 异常分数 + 原始能量暴增）"""
        ae_conf = 0.0
        # --- 指标1: 累积能量加速比（斜率短窗/长窗比） ---
        # 注意：此指标在 Phase 2 中可能因累积能量饱和而失效，不作为主要判断依据
        if 'ae_cumulative_energy' in features:
            current_energy = features['ae_cumulative_energy']
            slope = current_energy - self.prev_energy if self.prev_energy > 0 else 0.0
            self.prev_energy = current_energy
            self.energy_slopes.append(slope)
            if len(self.energy_slopes) > 500:
                self.energy_slopes.pop(0)
            if len(self.energy_slopes) >= 100:
                short_slopes = self.energy_slopes[-30:]
                long_slopes = self.energy_slopes[-100:]
                short_mean = float(np.mean(short_slopes))
                long_mean = float(np.mean(long_slopes))
                if long_mean > 1e-12:
                    accel_ratio = short_mean / long_mean
                    effective_threshold = max(self.ae_slope_ratio_threshold * 0.6, 1.2)
                    if accel_ratio > effective_threshold:
                        ae_conf = max(ae_conf, min((accel_ratio - effective_threshold) / 2.0, 0.8))
        # --- 指标2: AE 事件率突增（500点后参与，上限降低到 0.3，避免掩盖 spike 检测） ---
        if 'ae_event_rate' in features and self.total_points > 500:
            event_rate = features['ae_event_rate']
            if event_rate > 0.6:
                ae_conf = max(ae_conf, min((event_rate - 0.6) / 0.4, 0.3))
        # --- 指标3: AE 异常分数突增（500点后参与） ---
        if 'ae_anomaly_score' in features and self.total_points > 500:
            score = features['ae_anomaly_score']
            if score > 0.4:
                ae_conf = max(ae_conf, min((score - 0.4) / 0.6, 0.4))
        # ========== 修复6: 指标4 — 原始AE能量尖峰频率检测（主要判断依据） ==========
        # AE原始能量有稀疏但幅度极大的尖峰（比正常高1000-10万倍）
        # 统计每200点窗口内的尖峰次数
        if 'raw_ae_total_energy' in features and self.total_points > 500:
            raw_energy = features['raw_ae_total_energy']
            if not hasattr(self, '_ae_spike_log'):
                self._ae_spike_log = []
            log_e = np.log10(max(raw_energy, 1e-10))
            self._ae_spike_log.append(log_e)
            if len(self._ae_spike_log) > 500:
                self._ae_spike_log.pop(0)
            if len(self._ae_spike_log) >= 200:
                # 基线 = 最近200点的第30百分位数（排除尖峰影响）
                baseline = float(np.percentile(self._ae_spike_log, 30))
                # 尖峰阈值：超过基线 + 1.5 个对数单位（约30倍）
                spike_threshold = baseline + 1.5
                # 统计最近200点中超过阈值的点数
                spikes = sum(1 for v in self._ae_spike_log if v > spike_threshold)
                spike_rate = spikes / len(self._ae_spike_log)
                # 降低触发门槛：spike_rate > 0.03 即开始贡献置信度
                # 提高上限：最高可达 1.0（当 spike_rate >= 0.20 时）
                if spike_rate > 0.03:
                    ae_conf = max(ae_conf, min((spike_rate - 0.03) / 0.17, 1.0))
        return ae_conf

    def _detect_fo_drift(self, features):
        """检测光纤均值偏移（使用自适应阈值 + Z-score 标准化后的 FO）"""
        if 'fo_mean' not in features:
            return 0.0
        self.fo_warm_count += 1
        fo_mean = features['fo_mean']
        # ========== 修复5: 延长 FO warm-up 到 300 点 ==========
        if self.fo_warm_count <= 300:
            if self.fo_baseline_mean is None:
                self.fo_baseline_mean = fo_mean
            else:
                self.fo_baseline_mean = 0.995 * self.fo_baseline_mean + 0.005 * fo_mean
            return 0.0
        # FO 经过 Z-score 标准化，漂移信号以标准差为单位
        # ========== 修复5: 提高 FO 漂移检测阈值 ==========
        if abs(fo_mean) > self.fo_drift_threshold:
            return min((abs(fo_mean) - self.fo_drift_threshold) / 3.0, 1.0)
        # 同时也检测相对基线的偏移比例
        denom = max(abs(self.fo_baseline_mean), 1e-6)
        offset = abs(fo_mean - self.fo_baseline_mean) / denom
        if offset > self.fo_drift_threshold * 1.5:
            return min((offset - self.fo_drift_threshold * 1.5) / 1.0, 1.0)
        return 0.0

    def _decide_transition(self, strain_conf, ae_conf, fo_conf):
        """注意力机制动态权重融合：根据各源当前置信度动态调整权重"""
        self.total_points += 1
        # ========== 修复5: 强制最小稳定期 ==========
        # 前 min_stable_points 点不允许任何跃迁，确保基线充分建立
        if self.total_points <= self.min_stable_points:
            return self.current_phase
        if self.transition_cooldown > 0:
            self.transition_cooldown -= 1
            return self.current_phase
        # 基础权重（阶段先验）
        base_weights = {0: {'strain': 0.4, 'ae': 0.35, 'fo': 0.25},
                        1: {'strain': 0.25, 'ae': 0.50, 'fo': 0.25},
                        2: {'strain': 0.35, 'ae': 0.40, 'fo': 0.25}}
        base_w = base_weights.get(self.current_phase, base_weights[1])
        # ========== 注意力机制：根据各源置信度动态调整权重 ==========
        confs = {'strain': strain_conf, 'ae': ae_conf, 'fo': fo_conf}
        raw_attention = {k: max(v, 0.01) for k, v in confs.items()}
        total_att = sum(raw_attention.values())
        if total_att > 0:
            att_weights = {k: v / total_att for k, v in raw_attention.items()}
            mix_ratio = 0.4
            w = {}
            for k in base_w:
                w[k] = (1 - mix_ratio) * base_w[k] + mix_ratio * att_weights.get(k, 0)
            total_w = sum(w.values())
            if total_w > 0:
                w = {k: v / total_w for k, v in w.items()}
        else:
            w = base_w
        fusion_score = strain_conf * w['strain'] + ae_conf * w['ae'] + fo_conf * w['fo']
        # ========== 改造3: 使用自适应跃迁阈值 ==========
        threshold = self.transition_thresholds.get(self.current_phase, 0.50)
        # ========== 修复5: 每个阶段必须维持至少一定点数才能再次跃迁 ==========
        min_phase_duration = MIN_PHASE_DURATION  # 每个阶段至少维持 300 点
        points_in_phase = self.total_points - self.phase_entry_points.get(self.current_phase, 0)
        if (fusion_score > threshold and self.current_phase < 3
                and points_in_phase >= min_phase_duration):
            self.current_phase += 1
            self.phase_entry_points[self.current_phase] = self.total_points
            self.transition_cooldown = self.cooldown_period
            print(f'    [阶段跃迁] Phase {self.current_phase - 1} → Phase {self.current_phase} '
                  f'(融合置信度: {fusion_score:.3f}, '
                  f'应变:{strain_conf:.2f} AE:{ae_conf:.2f} FO:{fo_conf:.2f})')
        return self.current_phase

    def update(self, strain_val, features):
        # ========== 改造3: 在 update 中收集基线 ==========
        if not self.baseline_collected:
            self._collect_baseline(strain_val, features)
        if self.transition_cooldown > 0:
            self.transition_cooldown -= 1
        self._update_buffers(strain_val, features)
        strain_conf = self._detect_strain_jump(strain_val, features)
        ae_conf = self._detect_ae_transition(features)
        fo_conf = self._detect_fo_drift(features)
        phase = self._decide_transition(strain_conf, ae_conf, fo_conf)
        self.phase_history.append(phase)
        if len(self.debug_info) < 10000:
            self.debug_info.append({'strain_conf': strain_conf, 'ae_conf': ae_conf,
                                    'fo_conf': fo_conf, 'phase': phase})
        return phase

    def get_stages(self):
        return np.array(self.phase_history, dtype=int)
