#!/usr/bin/env python3
# Force–Velocity comparison helper
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================== 사용자 설정 ===================
file1 = "/home/yejin/ros2_ws/src/onelink_description/onelink_description/result/force_vel_K1.csv"
file2 = "/home/yejin/ros2_ws/src/onelink_description/onelink_description/result/force_vel_K3.csv"
# file1 = "/home/yejin/ros2_ws/src/onelink_description/onelink_description/result/force_vel_K2.csv"
# file2 = "/home/yejin/ros2_ws/src/onelink_description/onelink_description/result/force_vel_K4.csv"

sheet_name = None                 # 엑셀일 때만 의미 있음
time_window = (0.0, 30.0)         # <== 더 이상 사용하지 않음 (무시됨)
rolling_window = 0                # 스무딩 끄기: 0 또는 1
plot_every_n = 1                  # 샘플링 간격(1=모든점)
compute_velocity_from_position = False  # 속도 컬럼 없으면 True
assumed_sampling_hz = None        # 위치→속도 미분 시 샘플링(예: 100.0)
save_png = "force_velocity_compare.png"

# === 이상치(Outlier) 필터 설정 ===
# |force|가 이 값보다 크면 제거. 예: 750로 두면 +/-750 초과값 무시. 사용 안 하면 None
force_abs_max = 700            # 예) 750
# 분위수 범위로 남길 구간만 유지. 예: (0, 99.5)면 상위 0.5% 컷. 사용 안 하면 None
force_percentile_keep = (0.0, 99.5)
# ==================================================

COL_ALIASES = {
    "time": ("시간","time","Time","timestamp","Timestamp","t"),
    "position": ("위치","position","Position","pos","Pos","x","X"),
    "velocity": ("속도","velocity","Velocity","vel","Vel","v","V"),
    "force": ("힘","force","Force","F","Fx","fy","fz","wrench_fx"),
}

def _find_col(df, wanted):
    aliases = COL_ALIASES[wanted]
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns: return a
        if a.lower() in lower_map: return lower_map[a.lower()]
    for c in df.columns:
        for a in aliases:
            if a.lower() in str(c).lower(): return c
    return None

def _ensure_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def _compute_velocity_from_position(pos, time_s=None, fs=None):
    p = _ensure_numeric(pos).to_numpy(dtype=float)
    if time_s is not None and np.isfinite(time_s).any():
        t = _ensure_numeric(time_s).to_numpy(dtype=float)
        dt = np.diff(t); dt[dt <= 0] = np.nan
        v = np.empty_like(p); v[:] = np.nan
        v[1:] = np.diff(p) / dt
    else:
        if fs is None or fs <= 0:
            raise ValueError("Set a positive sampling rate if no time column is provided.")
        dt = 1.0 / fs
        v = np.gradient(p, dt)
    return pd.Series(v, index=pos.index)

def _load_table(path, sheet=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet)
        if isinstance(df, dict): df = next(iter(df.values()))
        return df
    # CSV/TXT: 구분자 자동탐지, 인코딩 BOM 대응
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

def _prep_one(path, sheet=None, time_window=None, rolling_window=5,
              compute_velocity_from_position=False, assumed_sampling_hz=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = _load_table(path, sheet=sheet)
    df = df.dropna(axis=1, how="all")

    c_time = _find_col(df, "time")
    c_pos  = _find_col(df, "position")
    c_vel  = _find_col(df, "velocity")
    c_for  = _find_col(df, "force")
    if c_for is None:
        raise ValueError(f"[{os.path.basename(path)}] Missing 'force/힘' column. Columns: {list(df.columns)}")

    out = pd.DataFrame()
    if c_time is not None:
        if np.issubdtype(df[c_time].dtype, np.datetime64):
            t0 = df[c_time].iloc[0]
            out["time_s"] = (df[c_time] - t0).dt.total_seconds()
        else:
            out["time_s"] = _ensure_numeric(df[c_time])
    else:
        out["time_s"] = np.nan

    if c_vel is not None:
        out["vel"] = _ensure_numeric(df[c_vel])
    elif compute_velocity_from_position:
        if c_pos is None:
            raise ValueError(f"[{os.path.basename(path)}] Need a position column to compute velocity.")
        out["vel"] = _compute_velocity_from_position(
            df[c_pos],
            out["time_s"] if np.isfinite(out["time_s"]).any() else None,
            assumed_sampling_hz
        )
    else:
        raise ValueError(f"[{os.path.basename(path)}] No velocity column. Set compute_velocity_from_position=True to derive from position.")

    out["force"] = _ensure_numeric(df[c_for])
    out = out.dropna(subset=["vel","force"])

    # ▼▼ time_window는 더 이상 사용하지 않음 (요청대로 무시) ▼▼
    # if time_window is not None and np.isfinite(out["time_s"]).any():
    #     t0, t1 = time_window
    #     out = out[(out["time_s"] >= t0) & (out["time_s"] <= t1)]

    if rolling_window and rolling_window > 1:
        out["vel"]   = out["vel"].rolling(rolling_window, min_periods=1, center=True).mean()
        out["force"] = out["force"].rolling(rolling_window, min_periods=1, center=True).mean()

    return out.reset_index(drop=True)

def _clip_to_common_tmax(A: pd.DataFrame, B: pd.DataFrame):
    """두 파일의 최대 시간 중 작은 값(t_max_common)까지만 사용."""
    if "time_s" not in A.columns or "time_s" not in B.columns:
        return A, B
    if not (np.isfinite(A["time_s"]).any() and np.isfinite(B["time_s"]).any()):
        return A, B
    tmax_common = min(np.nanmax(A["time_s"]), np.nanmax(B["time_s"]))
    A2 = A[A["time_s"] <= tmax_common].copy()
    B2 = B[B["time_s"] <= tmax_common].copy()
    return A2.reset_index(drop=True), B2.reset_index(drop=True)

def _filter_outliers(D: pd.DataFrame, name: str) -> pd.DataFrame:
    """힘 컬럼 기준 이상치 제거(절댓값 임계 + 분위수 범위)."""
    before = len(D)
    if force_abs_max is not None:
        D = D[D["force"].abs() <= float(force_abs_max)]
    if force_percentile_keep is not None and len(D) > 0:
        lo, hi = force_percentile_keep
        lo = max(0.0, float(lo)); hi = min(100.0, float(hi))
        if hi > lo:
            qlo, qhi = D["force"].quantile([lo/100.0, hi/100.0])
            D = D[(D["force"] >= qlo) & (D["force"] <= qhi)]
    removed = before - len(D)
    if removed > 0:
        print(f"[FILTER] {name}: removed {removed} outliers (kept {len(D)})")
    return D

def compare_force_velocity(path_a, path_b, sheet=None, time_window=None, rolling_window=5,
                           plot_every_n=3, compute_velocity_from_position=False,
                           assumed_sampling_hz=None, save_png="force_velocity_compare.png"):
    # time_window 전달되더라도 _prep_one에서 무시됨
    A = _prep_one(path_a, sheet, time_window, rolling_window, compute_velocity_from_position, assumed_sampling_hz)
    B = _prep_one(path_b, sheet, time_window, rolling_window, compute_velocity_from_position, assumed_sampling_hz)

    # === 공통 최대 시간까지만 사용 ===
    A, B = _clip_to_common_tmax(A, B)

    # === 이상치 제거 ===
    A = _filter_outliers(A, os.path.basename(path_a))
    B = _filter_outliers(B, os.path.basename(path_b))

    Ap = A.iloc[::max(1, plot_every_n)]
    Bp = B.iloc[::max(1, plot_every_n)]

    plt.figure(figsize=(8,6))
    plt.scatter(Ap["vel"], Ap["force"], s=12, label=os.path.basename(path_a), alpha=0.6)
    plt.scatter(Bp["vel"], Bp["force"], s=12, label=os.path.basename(path_b), alpha=0.6)

    def linfit(x, y):
        x = x.to_numpy(dtype=float); y = y.to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2: return None, None
        M = np.vstack([x[ok], np.ones(ok.sum())]).T
        a, b = np.linalg.lstsq(M, y[ok], rcond=None)[0]
        return a, b

    a1, b1 = linfit(A["vel"], A["force"])
    a2, b2 = linfit(B["vel"], B["force"])
    for (D, a, b, lbl) in [(A,a1,b1,os.path.basename(path_a)), (B,a2,b2,os.path.basename(path_b))]:
        if a is not None:
            vx = np.linspace(np.nanmin(D["vel"]), np.nanmax(D["vel"]), 100)
            vy = a*vx + b
            plt.plot(vx, vy, linewidth=2, label=f"{lbl} fit: F≈{a:.3g}·v + {b:.3g}")

    plt.xlabel("Velocity"); plt.ylabel("Force"); plt.title("Force–Velocity (clip to common t_max)")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(save_png, dpi=150, bbox_inches="tight"); plt.show()
    return os.path.abspath(save_png)

# ========= 파일 경로가 채워져 있으면 자동 실행 =========
if __name__ == "__main__":
    if file1 and file2:
        out = compare_force_velocity(
            file1, file2,
            sheet=sheet_name,
            time_window=None,  # <-- 어차피 무시되지만 명시적으로 None
            rolling_window=rolling_window,
            plot_every_n=plot_every_n,
            compute_velocity_from_position=compute_velocity_from_position,
            assumed_sampling_hz=assumed_sampling_hz,
            save_png=save_png
        )
        print("Saved figure to:", out)
    else:
        print("Set file1 and file2 variables and run again.")
