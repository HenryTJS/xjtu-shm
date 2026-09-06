# -*- coding: utf-8 -*-
"""T5 v6 / T7: 损伤度 D——组合证据（损伤型事件 + 全能量加速 + 应变(降权辅证)）

证据工程结论(2026-09-06):
  1. "损伤型事件"(能量显著超背景)证据在 AE 全程活跃组(012/013/023)早期干净:
     早期磨合/普通 AE 不算, 损伤型事件在 30-35% 才出现 → 解决 AE 磨合污染;
  2. 静默型(020/017)无巨事件, 需"全能量累积加速"证据 → e_ae = max(e_dmg, e_full);
  3. 应变单调漂移(载荷)与真发散同形态, 权重无法区分 → e_strain 降权(辅证)。

T7 参数化: 全部可调参数集中在 __init__ kwargs(默认=v6 配置)。
调用方(E/在线)不传参即默认; 校准脚本通过 params dict 注入。

证据:
  e_dmg    = 块内损伤型事件能量 log EWMA      (损伤型: logE>背景p60+LIFT)
  e_full   = 全能量累积 log 加速度(短/长窗斜率差 EWMA)
  e_ae     = max(e_dmg, e_full)
  e_strain = estrain_w × 应变 std 相对滚动低分位发散
  risk     = max(e_ae, e_strain)
  D        = 升快降慢累积(rise 追 risk, fall 回落)
"""
import numpy as np
from .config import EXT_BLOCK_PTS


class OnlineDamageIndex:
    def __init__(self, params=None):
        p = dict(params or {})
        # --- 证据级(可校准) ---
        self.bg_warm = int(p.get('bg_warm', 200))       # 背景建立事件数
        self.bg_max = int(p.get('bg_max', 800))         # 背景滚动窗上限
        self.lift = float(p.get('lift', 2.0))           # 损伤型判据: 背景p60+lift(log)
        self.e_dmg_scale = float(p.get('e_dmg_scale', 2.5))
        self.acc_scale = float(p.get('acc_scale', 0.02))
        self.estrain_w = float(p.get('estrain_w', 0.6))
        self.strain_ratio_gain = float(p.get('strain_ratio_gain', 2.0))
        self.estrain_min_blocks = int(p.get('estrain_min_blocks', 20))
        # --- D 状态机(可校准) ---
        self.rise = float(p.get('rise', 0.12))
        self.fall = float(p.get('fall', 0.008))
        self.levels = [0.25, 0.55, 0.85]
        # --- T8 消融模式 ---
        # full        完整 v6: e_ae=max(e_dmg,e_full); risk=max(e_ae,e_strain); 升快降慢
        # no_dmg      去损伤型事件分类(e_ae=e_full)
        # no_full     去全能量加速(e_ae=e_dmg)
        # no_strain   去应变证据(risk=e_ae)
        # only_strain 仅应变证据(risk=e_strain)
        # no_accum    去单调累积(D=risk, 无记忆)
        self.abl = str(p.get('abl', 'full'))
        # --- v7 结构: 磨合/加载抑制窗(点) ---
        # 前 suppress_pts 点内证据照常学习(基线/背景), 但 risk 不累积进 D(排除加载/磨合瞬态)。
        # 在线无总长时传固定点; 离线评估可传 n*frac。
        self._suppress_pts = int(p.get('suppress_pts', 0))
        # --- v8 结构: 在线就绪检测(route A, 磨合/加载结束 = AE能量峰值回落 AND 应变稳定) ---
        # ready_mode='off' 不用(纯 v6/v7); 'auto' 用在线检测替代寿命分数抑制。
        # 语义: 抑制期 = 就绪前(证据照常学基线, 但 risk 不累积); 就绪后永不抑制。
        self.ready_mode = str(p.get('ready_mode', 'off'))
        # v8b: 就绪 = 最近连续 ready_quiet_blk 块"平静"(无 AE 事件 且 应变 std 不高)
        #       → 加载/磨合/漂移过渡段活动高不平静; 进入稳定服役期才就绪(latch)。
        self.ready_quiet_blk = int(p.get('ready_quiet_blk', 8))
        self.ready_st_fac = float(p.get('ready_st_fac', 1.5))
        self.reset()

    def reset(self):
        self._bpts = 0
        self._seen_pts = 0
        self._block_dmg = 0.0        # 块损伤型能量
        self._block_all = 0.0        # 块全能量(事件)
        self._strain_vals = []
        self._bg_loge = []
        self._bg_ready = False
        self._dlog_s = 0.0           # 损伤型块 log EWMA
        self._cum_energy = 0.0
        self._cum_logs = []
        self._accel_ewma = 0.0
        self._strain_stds = []
        self.risk = 0.0
        self.damage = 0.0
        self._bcount = 0
        self.level = 0
        self._last_e_ae = 0.0
        self._last_e_strain = 0.0
        self._ready = False
        self._ae_ch = False
        self._st_ch = False
        self._quiet_run = 0
        self._recent_aelog = []
        self._aelog_this = 0.0
        self._has_st_this = False

    def _on_event(self, peak):
        peak = max(float(peak), 0.0)
        loge = np.log10(peak ** 2 + 1e-12)
        if self._bg_ready:
            if loge > float(np.percentile(self._bg_loge, 60)) + self.lift:
                self._block_dmg += peak ** 2        # 损伤型
                return
            self._bg_loge.append(loge)
        else:
            self._bg_loge.append(loge)
            if len(self._bg_loge) >= self.bg_warm:
                self._bg_ready = True
        if len(self._bg_loge) > self.bg_max:
            self._bg_loge.pop(0)

    def _block_settle(self):
        self._bcount += 1
        self._aelog_this = np.log10(self._block_all + 1.0)   # 本块 AE 事件能量(块内累积), 供就绪检测
        # --- e_dmg ---
        dlog = np.log10(self._block_dmg + 1.0)
        self._block_dmg = 0.0
        self._dlog_s = 0.7 * self._dlog_s + 0.3 * dlog
        e_dmg = float(min(max(self._dlog_s / self.e_dmg_scale, 0.0), 1.0))
        # --- e_full (全能量累积加速) ---
        self._cum_energy += self._block_all
        self._block_all = 0.0
        self._cum_logs.append(np.log10(self._cum_energy + 1.0))
        if len(self._cum_logs) > 40:
            self._cum_logs.pop(0)
        e_full = 0.0
        logs = self._cum_logs
        if len(logs) >= 14:
            sl = (logs[-1] - logs[-13]) / 12.0
            ss = (logs[-1] - logs[-4]) / 3.0
            self._accel_ewma = 0.6 * self._accel_ewma + 0.4 * (ss - sl)
            e_full = float(min(max(self._accel_ewma / self.acc_scale, 0.0), 1.0))
        # --- 按消融模式组合 e_ae ---
        if self.abl == 'no_dmg':
            self._last_e_ae = e_full
        elif self.abl == 'no_full':
            self._last_e_ae = e_dmg
        else:
            self._last_e_ae = max(e_dmg, e_full)
        # e_strain (降权)
        e_strain = 0.0
        self._has_st_this = bool(self._strain_vals)
        if self._strain_vals:
            self._strain_stds.append(float(np.std(self._strain_vals)))
            self._strain_vals = []
            if len(self._strain_stds) > 120:
                self._strain_stds.pop(0)
        if len(self._strain_stds) >= self.estrain_min_blocks:
            sbase = float(np.percentile(self._strain_stds, 50))
            if sbase > 1e-9:
                ratio = self._strain_stds[-1] / sbase
                e_strain = float(min(max((ratio - 1.0) / self.strain_ratio_gain, 0.0), 1.0))
                e_strain *= self.estrain_w
        self._last_e_strain = e_strain
        # --- 在线就绪检测(route A) ---
        if self.ready_mode != 'off':
            self._check_ready()
        # --- risk / D(按消融模式) ---
        if self.abl == 'only_strain':
            self.risk = e_strain
        elif self.abl == 'no_strain':
            self.risk = self._last_e_ae
        else:
            self.risk = max(self._last_e_ae, e_strain)
        suppressed = (self._seen_pts <= self._suppress_pts) or \
                     (self.ready_mode != 'off' and not self._ready)
        if suppressed:
            # 磨合/加载/就绪前抑制期: 证据照常学, 但不进入损伤度
            self.damage = 0.0
            self.level = 0
            return
        if self.abl == 'no_accum':
            self.damage = max(0.0, min(1.0, self.risk))
        else:
            if self.risk > self.damage:
                self.damage += self.rise * (self.risk - self.damage)
            else:
                self.damage -= self.fall * self.damage
            self.damage = max(0.0, min(1.0, self.damage))
        self._update_level()

    def _check_ready(self):
        """在线就绪检测 v8b(route A): 就绪 = 最近连续 ready_quiet_blk 块"平静"。
        平静块定义: 无显著 AE 事件能量(aelog<0.05) 且 (应变 std 不高于运行中位 ready_st_fac 倍
        —— 或本块无应变/应变史不足时不判活跃)。
        加载/磨合/漂移段活动高 → 不平静; 进入稳定服役期后连续平静 → ready(latch, 此后永不抑制)。
        应变数据全程缺失的组: 仅由 AE 通道决定(此时 AE 也是其唯一证据源)。
        """
        if self._ready:
            return
        if self._bcount < 6:
            return
        # --- 本块是否"活跃" ---
        ae_active = self._aelog_this > 0.05                       # 本块有显著 AE 事件能量
        st_active = False
        if self._has_st_this and len(self._strain_stds) >= 8:
            base = float(np.percentile(self._strain_stds, 50))
            if base > 0 and self._strain_stds[-1] > base * self.ready_st_fac:
                st_active = True                                  # 应变 std 显著高于运行中位
        active = ae_active or st_active
        # --- 连续平静计数 ---
        if active:
            self._quiet_run = 0
        else:
            self._quiet_run += 1
            if self._quiet_run >= self.ready_quiet_blk:
                self._ready = True

    def _update_level(self):
        lv = 0
        for i, th in enumerate(self.levels):
            if self.damage >= th:
                lv = i + 1
        self.level = lv

    def update(self, strain, ae_peak=None):
        self._bpts += 1
        self._seen_pts += 1
        if ae_peak is not None:
            peak = max(float(ae_peak), 0.0)
            self._block_all += peak ** 2
            self._on_event(peak)
        if strain is not None and not np.isnan(strain):
            self._strain_vals.append(strain)
        if self._bpts >= EXT_BLOCK_PTS:
            self._bpts = 0
            self._block_settle()
        return self.damage

    def get_level(self):
        return self.level
