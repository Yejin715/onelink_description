#!/usr/bin/env python3
# time_series_compare_4.py — v(t), a(t), Fx(t) 3-서브플롯 비교 (최대 4개 파일)
import os, math
import numpy as np
import pandas as pd
import matplotlib
# 창 없이 저장만 하려면 주석 해제
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================== 사용자 설정 ===================
# 비교할 파일 경로들 (2~4개)
file_paths = [
    "/home/yejin/Desktop/limit_K1.csv",
    "/home/yejin/Desktop/limit_K2.csv",
    "/home/yejin/Desktop/limit_K3.csv",
    "/home/yejin/Desktop/limit_K4.csv",
]

sheet_name = None                 # 엑셀인 경우 시트명 (없으면 None)
rolling_window = 0                # 스무딩 윈도우(0/1=끄기, 5~15 추천)
plot_every_n = 1                  # 다운샘플링 간격(1=모든점)
compute_velocity_from_position = False  # 속도 컬럼 없으면 True
assumed_sampling_hz = None        # 시간 컬럼 없을 때 미분용 샘플링 Hz (필요시 지정)
out_png = "time_series_compare_4.png"   # 저장 파일명
# ===================================================

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

def _load_table(path, sheet=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet)
        if isinstance(df, dict): df = next(iter(df.values()))
        return df
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

def _compute_velocity_from_position(pos, time_s=None, fs=None):
    p = _ensure_numeric(pos).to_numpy(dtype=float)
    if time_s is not None and np.isfinite(time_s).any():
        t = _ensure_numeric(time_s).to_numpy(dtype=float)
        dt = np.diff(t); dt[dt <= 0] = np.nan
        v = np.empty_like(p); v[:] = np.nan
        v[1:] = np.diff(p) / dt
    else:
        if fs is None or fs <= 0:
            raise ValueError("Provide sampling Hz to compute velocity without time column.")
        v = np.gradient(p, 1.0/fs)
    return pd.Series(v)

def _compute_accel_from_velocity(vel, time_s=None, fs=None):
    v = _ensure_numeric(vel).to_numpy(dtype=float)
    if time_s is not None and np.isfinite(time_s).any():
        t = _ensure_numeric(time_s).to_numpy(dtype=float)
        a = np.gradient(v, t)
    else:
        if fs is None or fs <= 0:
            raise ValueError("Provide sampling Hz to compute accel without time column.")
        a = np.gradient(v, 1.0/fs)
    return pd.Series(a)

def _prep_one(path, sheet=None, rolling_window=5,
              compute_velocity_from_position=False, assumed_sampling_hz=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = _load_table(path, sheet=sheet).dropna(axis=1, how="all")

    c_time = _find_col(df, "time")
    c_pos  = _find_col(df, "position")
    c_vel  = _find_col(df, "velocity")
    c_for  = _find_col(df, "force")
    if c_for is None:
        raise ValueError(f"[{os.path.basename(path)}] Missing 'force/힘' column. Columns: {list(df.columns)}")

    out = pd.DataFrame()

    # time_s (각 파일 시작을 0초로 정규화)
    if c_time is not None:
        if np.issubdtype(df[c_time].dtype, np.datetime64):
            t0 = df[c_time].iloc[0]
            out["time_s"] = (df[c_time] - t0).dt.total_seconds()
        else:
            out["time_s"] = _ensure_numeric(df[c_time])
            # 음수/역전 방지용으로 첫 값을 0 기준으로 맞추기 (선택)
            if np.isfinite(out["time_s"]).any():
                out["time_s"] = out["time_s"] - out["time_s"].iloc[0]
    else:
        out["time_s"] = np.nan  # 시간축이 없으면 인덱스로 표시

    # velocity
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

    # acceleration
    out["acc"] = _compute_accel_from_velocity(
        out["vel"],
        out["time_s"] if np.isfinite(out["time_s"]).any() else None,
        assumed_sampling_hz
    )

    # force
    out["force"] = _ensure_numeric(df[c_for])

    # drop NaNs
    out = out.dropna(subset=["vel","acc","force"])

    # smoothing
    if rolling_window and rolling_window > 1:
        out["vel"]   = out["vel"].rolling(rolling_window, min_periods=1, center=True).mean()
        out["acc"]   = out["acc"].rolling(rolling_window, min_periods=1, center=True).mean()
        out["force"] = out["force"].rolling(rolling_window, min_periods=1, center=True).mean()

    return out.reset_index(drop=True)

def _clip_all_to_common_tmax(dfs):
    """모든 파일의 time_s 최대값 중 최소값까지 자르기."""
    # time_s가 있는 DataFrame들만 고려
    tmaxes = []
    for D in dfs:
        if "time_s" in D.columns and np.isfinite(D["time_s"]).any():
            tmaxes.append(np.nanmax(D["time_s"]))
    if not tmaxes:
        return dfs  # 전부 시간축이 없으면 그대로 반환
    tmax_common = np.nanmin(tmaxes)
    out = []
    for D in dfs:
        if "time_s" in D.columns and np.isfinite(D["time_s"]).any():
            out.append(D[D["time_s"] <= tmax_common].reset_index(drop=True))
        else:
            out.append(D)
    return out

def compare_time_series_multi(paths, sheet=None, rolling_window=5,
                              compute_velocity_from_position=False, assumed_sampling_hz=None,
                              save_png="time_series_compare_4.png"):
    if not (2 <= len(paths) <= 4):
        raise ValueError("Provide between 2 and 4 file paths.")
    labels = [os.path.basename(p) for p in paths]

    dfs = [ _prep_one(p, sheet, rolling_window, compute_velocity_from_position, assumed_sampling_hz)
            for p in paths ]

    # 공통 최대 시간으로 클리핑
    dfs = _clip_all_to_common_tmax(dfs)

    # 다운샘플링
    dfs = [ D.iloc[::max(1, plot_every_n)].copy() for D in dfs ]

    # ==== 서브플롯 ====
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11, 9), sharex=True)
    ax_v, ax_a, ax_f = axes

    # x축: time_s가 있으면 시간, 아니면 인덱스
    has_time = all(("time_s" in D.columns) for D in dfs)
    x_label = "Time [s]" if has_time else "Sample index"

    # 각 서브플롯에 모든 파일을 그리기
    for D, lbl in zip(dfs, labels):
        x = D["time_s"] if has_time else D.index
        ax_v.plot(x, D["vel"],   label=lbl)
        ax_a.plot(x, D["acc"],   label=lbl)
        ax_f.plot(x, D["force"], label=lbl)

    ax_v.set_ylabel("Velocity");     ax_v.grid(True)
    ax_a.set_ylabel("Acceleration"); ax_a.grid(True)
    ax_f.set_ylabel("Force (Fx)");   ax_f.set_xlabel(x_label); ax_f.grid(True)

    # 범례는 위쪽 하나로 통합
    handles, leg_labels = ax_v.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=min(len(paths), 4), frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Time-Series Comparison (up to 4 files): v(t), a(t), Fx(t)", y=1.06)
    fig.tight_layout()
    fig.savefig(save_png, dpi=150, bbox_inches="tight")
    plt.show()
    return os.path.abspath(save_png)

# ========= 파일 경로가 채워져 있으면 자동 실행 =========
if __name__ == "__main__":
    out = compare_time_series_multi(
        file_paths,
        sheet=sheet_name,
        rolling_window=rolling_window,
        compute_velocity_from_position=compute_velocity_from_position,
        assumed_sampling_hz=assumed_sampling_hz,
        save_png=out_png
    )
    print("Saved figure to:", out)
