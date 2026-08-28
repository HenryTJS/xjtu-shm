# -*- coding: utf-8 -*-
"""
疲劳机多源监测数据对齐脚本（AE 网格模式）
处理范围：试件 006-027

对齐策略（以声发射事件为网格）：
1. 时间统一：所有数据采样率 10Hz。光纤/应变时间戳统一换算到秒
   （ts_to_seconds 自动识别 0.1s 计数单位并 ×0.1，无需按试件单独处理）
2. 数据清洗：跳过 015 文件头、剔除全空通道、处理 NaN
3. 以 AE 事件为时间网格：AE 无时间戳，行号均匀映射到实验时长 [0,T]
   （近似：各源时长相同、丢数据速率相同）
4. 光纤/应变线性插值到每个 AE 事件时刻；仅在各自有效覆盖范围内插值
   （范围外置 NaN，不外推伪造数据）
5. 输出：{sid}.csv —— 行数 = AE 事件数，AE 原始特征 100% 保留，
   光纤/应变逐 AE 事件对齐；另输出对齐元信息（含丢数据段检测）

用法：python align.py
"""
import os, warnings, traceback
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============ 配置 ============
ROOT = os.path.dirname(os.path.abspath(__file__))   # 当前工作区数据根（d:\lixiang）
OUT = os.path.join(ROOT, "aligned")   # 对齐输出直接放根目录 aligned\（不再嵌套 output\对齐数据）
OUT_D = OUT
os.makedirs(OUT_D, exist_ok=True)

SPECIMENS = [f"{i:03d}" for i in range(6, 28)]
# ==============================


def ts_to_seconds(t_series):
    """将时间戳统一换算为秒（返回与输入等长数组，保留 NaN）。

    所有数据采样率均为 10Hz：
    - 光纤 Time Stamp 相邻差≈1.0 → 0.1s 计数单位 → 需 ×0.1
    - 应变有两种：dt≈0.1 已是秒(10Hz)；dt≈1.0 为 0.1s 计数单位 → 需 ×0.1
    统一按中位间隔自动判断（>=0.2s 视为 0.1s 计数单位）。
    """
    t = pd.to_numeric(t_series, errors='coerce').values.astype(float)
    valid = t[~np.isnan(t)]
    if len(valid) < 2:
        return t
    dt = float(np.median(np.diff(valid)))
    if dt >= 0.2:
        return t * 0.1
    return t


def detect_gaps(time_s, mult=1.5, min_gap_s=0.5):
    """检测时间轴缺失段（丢数据）。

    time_s: 已换算为秒、已排序的时间戳数组
    返回 [(start_s, end_s, lost_s), ...]，lost_s 为扣除正常间隔后的缺失时长。
    """
    time_s = np.asarray(time_s, dtype=float)
    if len(time_s) < 3:
        return []
    d = np.diff(time_s)
    med = float(np.median(d))
    if med <= 0:
        return []
    gaps = []
    for i, g in enumerate(d):
        if g > mult * med and g >= min_gap_s:
            gaps.append((float(time_s[i]), float(time_s[i + 1]), float(g - med)))
    return gaps


def read_fiber(sid, sdir):
    """读取光纤数据 → DataFrame[time_s, fiber_1..fiber_k]"""
    files = [f for f in os.listdir(sdir) if "光纤" in f]
    if not files:
        return None, "无光纤文件"
    fp = os.path.join(sdir, files[0])
    skiprows = 3 if sid == "015" else 0

    if fp.endswith(".xlsx"):
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            return None, f"xlsx读取失败: {e}"
    else:
        df = None
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]:
            try:
                df = pd.read_csv(fp, encoding=enc, skiprows=skiprows)
                break
            except Exception:
                continue
        if df is None:
            return None, "CSV编码无法识别"

    cols = list(df.columns)
    if len(cols) < 2:
        return None, f"列数不足: {cols}"

    time_s = ts_to_seconds(df.iloc[:, 0])

    channels = {}
    for i, c in enumerate(cols[1:], 1):
        vals = pd.to_numeric(df[c], errors='coerce').values
        if np.sum(~np.isnan(vals)) > 10:
            channels[f"fiber_{i}"] = vals

    if not channels:
        return None, "无有效通道"

    result = pd.DataFrame({"time_s": time_s})
    for k, v in channels.items():
        result[k] = v
    result = result.dropna(subset=["time_s"])
    ch_cols = list(channels.keys())
    result = result.dropna(subset=ch_cols, how='all')
    result = result.sort_values("time_s").reset_index(drop=True)
    return result, None


def read_strain(sid, sdir):
    """读取应变数据 → DataFrame[time_s, strain]"""
    files = [f for f in os.listdir(sdir) if "应变" in f]
    if not files:
        return None, "无应变文件"
    fp = os.path.join(sdir, files[0])

    if fp.endswith(".xlsx"):
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            return None, f"xlsx读取失败: {e}"
    else:
        df = None
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]:
            try:
                df = pd.read_csv(fp, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            return None, "CSV编码无法识别"

    if df.shape[1] < 2:
        return None, f"列数不足: {df.shape}"

    time_s = ts_to_seconds(df.iloc[:, 0])
    v = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    result = pd.DataFrame({"time_s": time_s, "strain": v})
    result = result.dropna()
    result = result.sort_values("time_s").reset_index(drop=True)
    return result, None


def read_ae(sid, sdir):
    """读取声发射数据 → DataFrame"""
    files = [f for f in os.listdir(sdir) if "声发射" in f]
    if not files:
        return None, "无声发射文件"
    fp = os.path.join(sdir, files[0])
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            df = pd.read_csv(fp, encoding=enc)
            return df, None
        except Exception:
            continue
    return None, "CSV编码无法识别"


def detect_oscillation(strain_df):
    """检测应变信号是否振荡：信号本身零交叉率>30%（区分正负交替的循环应变与平滑应变）"""
    v = strain_df['strain'].values
    if len(v) < 100:
        return False
    sample = v[:min(len(v), 5000)]
    signs = np.sign(sample)
    signs[signs == 0] = 1
    zero_crossings = np.sum(signs[1:] != signs[:-1])
    return (zero_crossings / len(signs)) > 0.3


def align_sources_to_ae(ae_df, fb_df, ys_df, t_start, T):
    """以声发射事件为网格对齐多源数据。

    声发射没有时间戳，将其视为事件序列：行号均匀映射到实验时长 [0, T]
    （近似：各源时长相同、丢数据速率相同）。
    光纤/应变具有真实 10Hz 时间戳，线性插值到每个 AE 事件时刻；
    仅在各自有效覆盖范围内插值（范围外置 NaN，不外推伪造数据）。

    返回 DataFrame，行数 = AE 事件数，AE 原始特征 100% 保留。
    """
    if ae_df is None or len(ae_df) < 2:
        return None
    N = len(ae_df)
    ae_norm = np.linspace(0, 1, N)
    ae_t = np.linspace(t_start, t_start + T, N)

    out = pd.DataFrame({
        "ae_event": np.arange(N),
        "ae_time_s": ae_t,
        "ae_norm": ae_norm,
    })

    # ---- 完整保留 AE 原始特征（全部列） ----
    for c in ae_df.columns:
        out[f"ae_{c}"] = pd.to_numeric(ae_df[c], errors='coerce').values

    # ---- 光纤：线性插值到 AE 事件时刻 ----
    if fb_df is not None and len(fb_df) > 1:
        fb_t = fb_df['time_s'].values.astype(float)
        fb_norm = (fb_t - t_start) / T
        for c in [c for c in fb_df.columns if c.startswith("fiber_")]:
            v = pd.to_numeric(fb_df[c], errors='coerce').values.astype(float)
            valid = ~np.isnan(v) & ~np.isnan(fb_norm)
            if valid.sum() > 1:
                interp = np.interp(ae_norm, fb_norm[valid], v[valid])
                lo, hi = float(np.min(fb_norm[valid])), float(np.max(fb_norm[valid]))
                interp[(ae_norm < lo) | (ae_norm > hi)] = np.nan
                out[c] = interp
        fb_cols = [c for c in out.columns if c.startswith("fiber_") and c != "fiber_mean"]
        if fb_cols:
            out["fiber_mean"] = out[fb_cols].mean(axis=1)

    # ---- 应变：线性插值到 AE 事件时刻 ----
    if ys_df is not None and len(ys_df) > 1:
        ys_t = ys_df['time_s'].values.astype(float)
        ys_v = ys_df['strain'].values.astype(float)
        ys_norm = (ys_t - t_start) / T
        valid = ~np.isnan(ys_v) & ~np.isnan(ys_norm)
        if valid.sum() > 1:
            interp = np.interp(ae_norm, ys_norm[valid], ys_v[valid])
            lo, hi = float(np.min(ys_norm[valid])), float(np.max(ys_norm[valid]))
            interp[(ae_norm < lo) | (ae_norm > hi)] = np.nan
            out["strain"] = interp

    return out


def process_specimen(sid):
    """处理单个试件，返回(ae_grid, meta)或(None, error)"""
    sdir = os.path.join(ROOT, sid)
    meta = {"试件": sid, "问题": []}

    # ---- 读取 ----
    fb_df, fb_err = read_fiber(sid, sdir)
    ys_df, ys_err = read_strain(sid, sdir)
    ae_df, ae_err = read_ae(sid, sdir)

    if fb_err: meta["问题"].append(f"光纤: {fb_err}")
    if ys_err: meta["问题"].append(f"应变: {ys_err}")
    if ae_err: meta["问题"].append(f"声发射: {ae_err}")

    if fb_df is None and ys_df is None:
        return None, "光纤和应变均缺失，无法确定时间轴"

    # ---- 确定实验时长 T ----
    # 以应变为主要时间基准（连续采集、覆盖完整实验）；无应变时以光纤为准
    # 这样可正确处理026这类光纤提前中断但应变持续到断裂的情况
    if ys_df is not None and len(ys_df) > 0:
        t_start = float(ys_df['time_s'].iloc[0])
        t_end = float(ys_df['time_s'].iloc[-1])
        time_ref = "应变"
    elif fb_df is not None and len(fb_df) > 0:
        t_start = float(fb_df['time_s'].iloc[0])
        t_end = float(fb_df['time_s'].iloc[-1])
        time_ref = "光纤"
    else:
        return None, "光纤和应变均缺失，无法确定时间轴"
    T = t_end - t_start
    meta["实验时长_s"] = round(T, 1)
    meta["时间起点_s"] = round(t_start, 1)
    meta["时间基准"] = time_ref

    # ---- 丢数据检测（时间戳缺口） ----
    for tag, df in (("应变", ys_df), ("光纤", fb_df)):
        if df is not None and len(df) > 1:
            gs = detect_gaps(df['time_s'].values)
            if gs:
                meta[f"{tag}缺失段"] = [f"{a:.1f}-{b:.1f}s(缺{c:.1f}s)" for a, b, c in gs]
                meta[f"{tag}缺失总时长_s"] = round(float(sum(c for _, _, c in gs)), 1)

    # 各源覆盖时长对比（检测中断/多录）
    if fb_df is not None and len(fb_df) > 0 and ys_df is not None and len(ys_df) > 0:
        fb_dur = float(fb_df['time_s'].iloc[-1] - fb_df['time_s'].iloc[0])
        if fb_dur > T * 1.1:
            meta["问题"].append(f"光纤时长{fb_dur:.0f}s 明显长于应变{T:.0f}s，应变可能提前结束")
        elif fb_dur < T * 0.9:
            meta["问题"].append(f"光纤仅覆盖{fb_dur/T*100:.0f}%实验时长，可能中途中断")

    # ---- 元信息 ----
    meta["光纤通道数"] = len([c for c in fb_df.columns if c.startswith("fiber_")]) if fb_df is not None else 0
    if ys_df is not None and len(ys_df) > 0:
        meta["应变振荡"] = detect_oscillation(ys_df)
    meta["声发射行数"] = len(ae_df) if ae_df is not None else 0

    # ---- 以声发射事件为网格对齐 ----
    ae_grid = align_sources_to_ae(ae_df, fb_df, ys_df, t_start, T)
    if ae_grid is None:
        meta["问题"].append("声发射数据不足，无法生成AE网格对齐输出")
        return ae_grid, meta

    meta["输出列"] = list(ae_grid.columns)
    meta["数据完整率"] = round(ae_grid.notna().mean().mean(), 3)
    return ae_grid, meta


# ============ 主流程 ============
if __name__ == "__main__":
    all_meta = []
    failed = []

    for sid in SPECIMENS:
        try:
            ae_grid, meta = process_specimen(sid)
            if ae_grid is None:
                print(f"[跳过] {sid}: {meta.get('问题', ['未知'])}")
                failed.append((sid, str(meta.get("问题", []))))
                all_meta.append(meta)
                continue
            # 保存 AE 网格对齐（行数 = AE 事件数，AE 原始特征 100% 保留）
            ae_path = os.path.join(OUT_D, f"{sid}.csv")
            ae_grid.to_csv(ae_path, index=False, encoding="utf-8-sig", float_format="%.6f")
            all_meta.append(meta)
            n_issues = len(meta.get("问题", []))
            miss = meta.get("应变缺失总时长_s", 0) + meta.get("光纤缺失总时长_s", 0)
            print(f"[完成] {sid}: T={meta['实验时长_s']:.0f}s, AE={meta['声发射行数']}行, "
                  f"光纤{meta['光纤通道数']}ch, 问题={n_issues}, 缺失={miss}s")
        except Exception as e:
            traceback.print_exc()
            failed.append((sid, str(e)))
            print(f"[异常] {sid}: {e}")

    # 保存元信息（CSV 已含全部字段，json 副本无读取方，不再生成）
    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(os.path.join(OUT, "对齐元信息.csv"), index=False, encoding="utf-8-sig")

    print(f"\n===== 对齐完成（AE 网格模式） =====")
    print(f"成功: {len(all_meta) - len(failed)}/{len(SPECIMENS)}")
    print(f"失败/跳过: {len(failed)}")
    for sid, err in failed:
        print(f"  {sid}: {err}")
    print(f"输出目录: {OUT_D}")
