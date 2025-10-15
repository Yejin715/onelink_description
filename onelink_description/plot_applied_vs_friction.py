#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Applied–Friction comparison helper (Gazebo vs RecurDyn) + Outlier & Filter
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================== 사용자 설정 ===================
# (1) 우리 CSV (friction이 없으면 joint effort에서 유도 가능)
Gazebo_file    = '/tmp/effort_K_3.csv'
# (2) 교수님 CSV (한 파일에 다음 컬럼들이 함께 존재한다고 가정)
RecurDyn_file = "/home/yejin/ros2_ws/src/onelink_description/onelink_description/result/recurdyn_force_data.csv"

# 스무딩(rolling mean) 창 크기 (0 또는 1 이면 미적용)
rolling_window = 0

# n개마다 하나씩만 그리기 (점 너무 많을 때 간소화)
plot_every_n = 1

# 우리 CSV에 friction 컬럼이 없다면 joint effort에서 유도하기
use_joint_effort_for_Gazebo = False     # True면 fric = -joint_effort 로 계산
flip_Gazebo_friction_sign    = False     # 센서 부호가 반작용(-)이면 True로 뒤집기

# 저장 파일명
save_png = "applied_vs_friction_compare.png"

# === 이상치(Outlier) 1차 제거 설정 ===
# |friction|이 이 값보다 크면 제거 (None이면 사용 안 함)
fric_abs_max = None
# 분위수 범위로 남길 구간만 유지. 예: (0, 99.5)면 상위 0.5% 컷 (None이면 사용 안 함)
fric_percentile_keep = None  # (0.0, 99.5) 등으로 사용 가능

# === 필터 설정 (이상치 제거 이후 2차 스무딩/필터) ============================
# kind in {"none","median","ema","savgol","butter"}
filter_kind = "median"
# 공통/개별 파라미터
median_window = 21          # 홀수 권장
ema_alpha     = 0.2
savgol_window = 21         # 홀수
savgol_poly   = 3
butter_cutoff_hz = 5.0     # Hz, 시간열에서 샘플레이트 자동 추정
butter_order     = 4
# Hampel spike repair(선택): True면 중앙값+MAD로 단발성 스파이크를 중앙값으로 교체
use_hampel_repair = False
hampel_window     = 11
hampel_n_sigma    = 3.0
# ============================================================================

# === 기준선(±μmg) 표시 ===
mu   = 0.5
mass = 280.85
g    = 9.81
# ==================================================

# 컬럼 별칭
COL_ALIASES = {
    "time": (
        "time [s]", "time", "Time", "timestamp", "Timestamp", "t", "시간"
    ),
    "applied": (
        "applied force", "cmd_f", "force", "effort", "F", "Fx"
    ),
    "friction": (
        "friction force", "friction", "wrench_fx",
        "friction force (static)",
        "friction force (static, kinetic)"
    ),
    "sensorf": (
        "sensor force"
    ),
    "effort": ("effort", "joint effort"),
}

def _find_col(df, wanted_key, candidates=None):
    if candidates is None:
        candidates = COL_ALIASES.get(wanted_key, ())
    cols = list(df.columns)
    lowmap = {str(c).strip().lower(): c for c in cols}
    for c in candidates:  # 완전/대소문자 무시 일치
        if c in cols:
            return c
        cl = str(c).strip().lower()
        if cl in lowmap:
            return lowmap[cl]
    # 부분 포함
    for c in cols:
        cl = str(c).strip().lower()
        for target in candidates:
            if str(target).strip().lower() in cl:
                return c
    return None

def _ensure_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def _load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        if isinstance(df, dict):
            df = next(iter(df.values()))
        return df
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

def _infer_time_col(df):
    for c in COL_ALIASES["time"]:
        if c in df.columns:
            return c
    return None

def _infer_fs(df):
    tcol = _infer_time_col(df)
    if tcol is None:
        return None, None
    t = _ensure_numeric(df[tcol]).to_numpy()
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return tcol, None
    fs = 1.0 / np.median(dt)
    return tcol, fs

def _smooth_series(s, win):
    if not win or win <= 1: return s
    return s.rolling(win, min_periods=1, center=True).mean()

def _hampel_repair(y: pd.Series, window=11, n_sigma=3.0) -> pd.Series:
    k = window // 2
    x = y.to_numpy().astype(float)
    out = x.copy()
    for i in range(len(x)):
        i0, i1 = max(0, i-k), min(len(x), i+k+1)
        win = x[i0:i1]
        med = np.nanmedian(win)
        mad = 1.4826 * np.nanmedian(np.abs(win - med))
        if not np.isfinite(mad) or mad == 0:
            continue
        if np.isfinite(x[i]) and abs(x[i] - med) > n_sigma * mad:
            out[i] = med
    return pd.Series(out, index=y.index)

def _apply_filter(df, series: pd.Series, ref_df=None) -> pd.Series:
    """filter_kind에 따라 series를 필터링.
       ref_df는 시간/샘플레이트 추정용(Butterworth에서 사용)."""
    s = series.copy()
    if use_hampel_repair:
        s = _hampel_repair(s, window=hampel_window, n_sigma=hampel_n_sigma)

    kind = (filter_kind or "none").lower()
    if kind in ("none", ""):
        pass
    elif kind == "median":
        s = s.rolling(median_window, center=True, min_periods=1).median()
    elif kind == "ema":
        s = s.ewm(alpha=ema_alpha, adjust=False).mean()
    elif kind == "savgol":
        from math import ceil
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            raise ImportError("Savitzky–Golay 필터를 쓰려면 scipy가 필요합니다: pip install scipy")
        w = savgol_window + (1 - savgol_window % 2)  # 홀수 강제
        arr = s.to_numpy()
        mask = np.isfinite(arr)
        out = arr.copy()
        if mask.sum() > w:
            out[mask] = savgol_filter(arr[mask], window_length=w, polyorder=savgol_poly, mode='interp')
        s = pd.Series(out, index=s.index)
    elif kind == "butter":
        try:
            from scipy.signal import butter, filtfilt
        except ImportError:
            raise ImportError("Butterworth 필터를 쓰려면 scipy가 필요합니다: pip install scipy")
        # 샘플레이트 추정
        tcol, fs = _infer_fs(ref_df if ref_df is not None else df)
        if fs is None:
            raise ValueError("시간열(time/t)에서 샘플레이트를 추정할 수 없어 Butterworth 적용 불가")
        wn = float(butter_cutoff_hz) / (0.5 * fs)
        if not (0 < wn < 1):
            raise ValueError(f"butter_cutoff_hz={butter_cutoff_hz}가 나이퀴스트(fs/2≈{fs/2:.3f} Hz)보다 작아야 합니다.")
        b, a = butter(butter_order, wn, btype='low', analog=False)
        arr = s.to_numpy()
        mask = np.isfinite(arr)
        out = arr.copy()
        if mask.sum() > (butter_order + 2):
            out[mask] = filtfilt(b, a, arr[mask])
        s = pd.Series(out, index=s.index)
    else:
        raise ValueError(f"Unknown filter_kind={filter_kind}")
    return s

def _filter_outliers_series(y: pd.Series, name: str) -> pd.Series:
    keep = pd.Series(True, index=y.index)
    if fric_abs_max is not None:
        keep &= (y.abs() <= float(fric_abs_max))
    if fric_percentile_keep is not None:
        lo, hi = fric_percentile_keep
        lo = max(0.0, float(lo)); hi = min(100.0, float(hi))
        if hi > lo and y.notna().sum() > 0:
            qlo, qhi = y.quantile([lo/100.0, hi/100.0])
            keep &= (y >= qlo) & (y <= qhi)
    removed = (~keep).sum()
    if removed > 0:
        print(f"[FILTER-OUTLIER] {name}: removed {removed} outliers (kept {keep.sum()})")
    return y.where(keep)

# -------------------- 데이터 준비 --------------------
def _prep_Gazebo_old(path, use_joint_effort=False, flip=False, rolling_window=0, label=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[Gazebo] File not found: {path}")
    df = _load_table(path).dropna(axis=1, how="all")

    c_applied = _find_col(df, "applied")
    if c_applied is None:
        raise ValueError(f"[Gazebo] Missing 'applied force' column. Columns: {list(df.columns)}")
    x = _ensure_numeric(df[c_applied])

    if use_joint_effort:
        c_eff = _find_col(df, "effort")
        if c_eff is None:
            raise ValueError(f"[Gazebo] Need 'effort' column for use_joint_effort=True")
        y = -_ensure_numeric(df[c_eff])   # 반작용 가정
    else:
        c_fric = _find_col(df, "friction", candidates=(
            "friction force (static, kinetic)",
            "friction force",
            "friction", "wrench_fx"
        ))
        if c_fric is None:
            raise ValueError(f"[Gazebo] Missing friction column. Add one or set use_joint_effort=True")
        y = _ensure_numeric(df[c_fric])

    if flip:
        y = -y

    # 1) 이상치 컷
    y = _filter_outliers_series(y, name=os.path.basename(path))
    x = x.where(y.notna()); y = y.dropna(); x = x.dropna()

    # 2) 선택형 필터(적용 대상: x와 y 둘 다)
    # x = _apply_filter(df, x, ref_df=df)
    # y = _apply_filter(df, y, ref_df=df)

    # 3) (선택) 롤링 스무딩 추가
    x = _smooth_series(x, rolling_window)
    y = _smooth_series(y, rolling_window)

    if plot_every_n and plot_every_n > 1:
        x = x.iloc[::plot_every_n]
        y = y.iloc[::plot_every_n]

    return x, y, (label or f"Gazebo ({'joint_effort' if use_joint_effort else 'fric'})")

def _prep_Gazebo(path, use_joint_effort=False, flip=False, rolling_window=0, label=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[Gazebo] File not found: {path}")
    df = _load_table(path).dropna(axis=1, how="all")

    c_applied = _find_col(df, "applied")
    if c_applied is None:
        raise ValueError(f"[Gazebo] Missing 'applied force' column. Columns: {list(df.columns)}")
    x = _ensure_numeric(df[c_applied])

    if use_joint_effort:
        c_eff = _find_col(df, "effort")
        if c_eff is None:
            raise ValueError(f"[Gazebo] Need 'effort' column for use_joint_effort=True")
        y = -_ensure_numeric(df[c_eff])   # 반작용 가정
    else:
        c_fric = _find_col(df, "friction", candidates=(
            "friction force (static, kinetic)",
            "friction force",
            "friction", "wrench_fx"
        ))
        if c_fric is None:
            raise ValueError(f"[Gazebo] Missing friction column. Add one or set use_joint_effort=True")
        y = _ensure_numeric(df[c_fric])

    if flip:
        y = -y

    # 1) 이상치 컷
    y = _filter_outliers_series(y, name=os.path.basename(path))
    x = x.where(y.notna()); y = y.dropna(); x = x.dropna()

    # 2) 선택형 필터(적용 대상: x와 y 둘 다)
    x = _apply_filter(df, x, ref_df=df)
    y = _apply_filter(df, y, ref_df=df)

    # 3) (선택) 롤링 스무딩 추가
    x = _smooth_series(x, rolling_window)
    y = _smooth_series(y, rolling_window)

    if plot_every_n and plot_every_n > 1:
        x = x.iloc[::plot_every_n]
        y = y.iloc[::plot_every_n]

    return x, y, (label or f"Gazebo ({'joint_effort' if use_joint_effort else 'fric'})")

def _prep_RecurDyn(path, rolling_window=0, base_label="RecurDyn"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[RecurDyn] File not found: {path}")
    df = _load_table(path).dropna(axis=1, how="all")

    c_applied = _find_col(df, "applied", candidates=("applied force", "cmd_f", "force"))
    if c_applied is None:
        raise ValueError(f"[RecurDyn] Missing 'applied force' column. Columns: {list(df.columns)}")
    x_raw = _ensure_numeric(df[c_applied])
    x_raw = _apply_filter(df, x_raw, ref_df=df)
    x_raw = _smooth_series(x_raw, rolling_window)

    c_static = _find_col(df, "friction", candidates=("friction force (static)",))
    c_snk    = _find_col(df, "friction", candidates=("friction force (static, kinetic)",))

    out = []

    if c_static is not None:
        ys = _ensure_numeric(df[c_static])
        ys = _filter_outliers_series(ys, name=f"{os.path.basename(path)}:static")
        xs = x_raw.where(ys.notna()); ys = ys.dropna(); xs = xs.dropna()
        ys = _apply_filter(df, ys, ref_df=df)
        xs = _apply_filter(df, xs, ref_df=df)
        xs = _smooth_series(xs, rolling_window)
        ys = _smooth_series(ys, rolling_window)
        if plot_every_n and plot_every_n > 1:
            xs = xs.iloc[::plot_every_n]; ys = ys.iloc[::plot_every_n]
        out.append((xs, ys, f"{base_label} (static)"))

    if c_snk is not None:
        yk = _ensure_numeric(df[c_snk])
        yk = _filter_outliers_series(yk, name=f"{os.path.basename(path)}:static+kinetic")
        xk = x_raw.where(yk.notna()); yk = yk.dropna(); xk = xk.dropna()
        yk = _apply_filter(df, yk, ref_df=df)
        xk = _apply_filter(df, xk, ref_df=df)
        xk = _smooth_series(xk, rolling_window)
        yk = _smooth_series(yk, rolling_window)
        if plot_every_n and plot_every_n > 1:
            xk = xk.iloc[::plot_every_n]; yk = yk.iloc[::plot_every_n]
        out.append((xk, yk, f"{base_label} (static+kinetic)"))

    if not out:
        raise ValueError(f"[RecurDyn] No friction series found (need 'friction force (static)' and/or 'friction force (static, kinetic)')")

    return out

# -------------------- 플로팅 --------------------
def plot_applied_vs_friction(Gazebo_path, RecurDyn_path, save_png="applied_vs_friction_compare.png"):
    xse, yse, lse = _prep_Gazebo_old(
        Gazebo_path,
        use_joint_effort=use_joint_effort_for_Gazebo,
        flip=flip_Gazebo_friction_sign,
        rolling_window=rolling_window,
        label="Gazebo (Static no filter)"
    )
    xo, yo, lo = _prep_Gazebo(
        Gazebo_path,
        use_joint_effort=use_joint_effort_for_Gazebo,
        flip=flip_Gazebo_friction_sign,
        rolling_window=rolling_window,
        label="Gazebo (Static)"
    )
    RecurDyn_series = _prep_RecurDyn(
        RecurDyn_path,
        rolling_window=rolling_window,
        base_label="RecurDyn"
    )

    plt.figure(figsize=(9,6))
    plt.plot(xo, yo, linewidth=7, alpha=1, label=lo, color="#9467BD", zorder=2)

    # 2) RecurDyn 선들: zorder=2~3
    sizes = [4, 1]
    colors_list = ["#d62728", "#2ca02c"]
    for i, (xs, ys, lb) in enumerate(RecurDyn_series):
        plt.plot(xs, ys, linestyle='-', linewidth=sizes[min(i, len(sizes)-1)],
                alpha=1, label=lb, color=colors_list[min(i, len(colors_list)-1)],
                zorder=3)

    # 3) 기준선: 제일 아래
    F_c = mu * mass * g
    plt.axhline(+F_c, linestyle='--', linewidth=1, label=f'+μmg ≈ {F_c:.1f} N', zorder=1, color="#6e6e6e")
    plt.axhline(-F_c, linestyle='--', linewidth=1, label=f'-μmg ≈ {-F_c:.1f} N', zorder=1, color="#6e6e6e")

    # 4) 노란 점: 제일 위 + 점 키우기 + 테두리
    plt.scatter(xse, yse, s=2, alpha=0.5, label=lse, color="#F0E442", linewidths=0.3, zorder=5)

    plt.xlabel("Applied Force [N]")
    plt.ylabel("Friction Force [N]")
    plt.ylim([-5000, 5000])
    plt.title(f"Friction vs Applied (filter={filter_kind})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_png, dpi=180, bbox_inches="tight")
    plt.show()
    return os.path.abspath(save_png)

# ========= 파일 경로가 채워져 있으면 자동 실행 =========
if __name__ == "__main__":
    if Gazebo_file and RecurDyn_file:
        out = plot_applied_vs_friction(Gazebo_file, RecurDyn_file, save_png=save_png)
        print("Saved figure to:", out)
    else:
        print("Set Gazebo_file and RecurDyn_file and run again.")
