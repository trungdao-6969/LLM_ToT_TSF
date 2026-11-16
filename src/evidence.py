"""
src/evidence.py  –  GĐ6, bước 6.1: tóm tắt dữ liệu tại origin d thành "bằng chứng" (x) cho LLM.

Nguyên tắc:
  1. CHỈ dùng dữ liệu trước origin (đến 23:00 ngày D-1), trừ lịch (thứ, ngày lễ) vốn biết trước.
     Ở L2/L3 được thêm nhiệt độ ngày D (dự báo / thật), đúng như mô hình được dùng.
  2. Chỉ đưa CON SỐ, không đưa kết luận diễn giải -> việc suy luận là của LLM.
  3. Ẩn danh: không có năm, không có tên bộ dữ liệu, phụ tải chỉ ở dạng tỉ số / phần trăm
     (không lộ quy mô) -> giảm nguy cơ LLM "nhớ" GEFCom2012 (data contamination).
  4. Mỗi thống kê gắn với ít nhất một nhóm giả thuyết (category) trong thiết kế 1.6:
     stable_seasonal, level_shift, temp_sensitive, calendar_anomaly, trend_change.
"""
import numpy as np
import pandas as pd

# Mô tả ngắn của từng thống kê, in kèm trong prompt để LLM hiểu con số
DESCRIPTIONS = {
    # --- mức tải, đổi chế độ (level_shift, trend_change)
    "level_7d_vs_365d": "mean load of last 7 days / mean load of last 365 days",
    "level_7d_vs_prev28d": "mean load of last 7 days / mean load of the 28 days before them",
    "level_28d_vs_lastyear": "mean load of last 28 days / same 28 calendar days one year earlier",
    "max_daily_jump_28d_pct": "largest day-to-day change of daily mean load in last 28 days (%)",
    "days_since_jump_over_20pct": "days since daily mean load last changed by >20% (none = no such jump in 90 days)",
    # --- độ nhiễu
    "cv_daily_load_28d_pct": "coefficient of variation of daily mean load over last 28 days (%)",
    "near_zero_hours_28d": "number of hours in last 28 days with load < 5% of the 28-day mean (outages)",
    # --- mùa vụ ngày/tuần (stable_seasonal)
    "profile_corr_last2weeks": "correlation of average 24h load profile: last week vs the week before",
    "naive_lag24_mape_14d_pct": "MAPE of 'same hour yesterday' forecast over last 14 days (%)",
    "naive_lag168_mape_14d_pct": "MAPE of 'same hour last week' forecast over last 14 days (%)",
    # --- nhiệt độ (temp_sensitive)
    "temp_mean_D1_F": "mean temperature of yesterday (°F)",
    "temp_max_D1_F": "max temperature of yesterday (°F)",
    "temp_change_D1_vs_D2_F": "mean temperature yesterday minus the day before (°F)",
    "temp_anomaly_7d_F": "mean temperature of last 7 days minus normal for this time of year (°F)",
    "corr_load_temp_28d": "correlation between daily mean load and daily mean temperature, last 28 days",
    "temp_mean_D_F": "mean temperature of the target day (forecast at L2, actual at L3; absent at L1)",
    # --- lịch (calendar_anomaly)
    "season": "season of the target day",
    "target_day_of_week": "day of week of the target day",
    "target_is_holiday": "target day is a public holiday",
    "holiday_in_last_7d": "a public holiday occurred in the last 7 days",
    "days_to_next_holiday": "days from target day to next public holiday (0 = target day)",
    # --- lượng dữ liệu (cho các núm train_years)
    "history_years_available": "years of hourly history available before the target day",
}

SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def _mape(y, yhat):
    ok = (~np.isnan(y)) & (~np.isnan(yhat)) & (y > 0)
    return float(np.mean(np.abs(y[ok] - yhat[ok]) / y[ok]) * 100)


def evidence(df, origin, temp_level="L1", temp_fc=None):
    """Trả về dict {tên thống kê: giá trị} tại origin (00:00 ngày D)."""
    origin = pd.Timestamp(origin)
    past = df.loc[:origin - pd.Timedelta(hours=1)]              # chỉ quá khứ
    load, T = past["load"], past["T"]
    daily = past[["load", "T"]].resample("D").mean()
    dl, dT = daily["load"], daily["T"]
    ev = {}

    # --- mức tải, đổi chế độ
    m7 = dl.iloc[-7:].mean()
    ev["level_7d_vs_365d"] = m7 / dl.iloc[-365:].mean()
    ev["level_7d_vs_prev28d"] = m7 / dl.iloc[-35:-7].mean()
    ly = dl.loc[origin - pd.Timedelta(days=365 + 28): origin - pd.Timedelta(days=366)]
    ev["level_28d_vs_lastyear"] = dl.iloc[-28:].mean() / ly.mean() if len(ly) >= 20 else None
    jump = (dl.pct_change().abs() * 100)
    ev["max_daily_jump_28d_pct"] = jump.iloc[-28:].max()
    big = jump.iloc[-90:][jump.iloc[-90:] > 20]
    ev["days_since_jump_over_20pct"] = (origin - big.index[-1]).days if len(big) else "none"

    # --- độ nhiễu
    ev["cv_daily_load_28d_pct"] = dl.iloc[-28:].std() / dl.iloc[-28:].mean() * 100
    l28 = load.iloc[-28 * 24:]
    ev["near_zero_hours_28d"] = int((l28 < 0.05 * l28.mean()).sum())

    # --- mùa vụ ngày/tuần
    prof = load.iloc[-14 * 24:].values.reshape(14, 24)
    ev["profile_corr_last2weeks"] = np.corrcoef(prof[7:].mean(0), prof[:7].mean(0))[0, 1]
    y14 = load.iloc[-14 * 24:].values
    ev["naive_lag24_mape_14d_pct"] = _mape(y14, load.shift(24).iloc[-14 * 24:].values)
    ev["naive_lag168_mape_14d_pct"] = _mape(y14, load.shift(168).iloc[-14 * 24:].values)

    # --- nhiệt độ
    ev["temp_mean_D1_F"] = dT.iloc[-1]
    ev["temp_max_D1_F"] = T.iloc[-24:].max()
    ev["temp_change_D1_vs_D2_F"] = dT.iloc[-1] - dT.iloc[-2]
    doy = dT.index.dayofyear
    target_doy = origin.dayofyear
    dist = np.minimum(np.abs(doy - target_doy), 365 - np.abs(doy - target_doy))
    clim = dT[(dist <= 7) & (dT.index < origin - pd.Timedelta(days=300))]   # cùng thời điểm các năm trước
    ev["temp_anomaly_7d_F"] = dT.iloc[-7:].mean() - clim.mean() if len(clim) else None
    ev["corr_load_temp_28d"] = np.corrcoef(dl.iloc[-28:], dT.iloc[-28:])[0, 1]
    if temp_level == "L3":
        ev["temp_mean_D_F"] = df.loc[origin: origin + pd.Timedelta(hours=23), "T"].mean()
    elif temp_level == "L2" and temp_fc is not None:
        ev["temp_mean_D_F"] = temp_fc.loc[origin: origin + pd.Timedelta(hours=23), "T_fc"].mean()

    # --- lịch (biết trước)
    hol_days = pd.DatetimeIndex(df.index[df["is_holiday"] == 1].normalize().unique())
    ev["season"] = SEASONS[origin.month]
    ev["target_day_of_week"] = origin.day_name()
    ev["target_is_holiday"] = bool(origin in hol_days)
    ev["holiday_in_last_7d"] = bool(((hol_days >= origin - pd.Timedelta(days=7)) & (hol_days < origin)).any())
    nxt = hol_days[hol_days >= origin]
    ev["days_to_next_holiday"] = int((nxt.min() - origin).days) if len(nxt) else "none"

    # --- lượng dữ liệu
    ev["history_years_available"] = (origin - df.index.min()).days / 365.25

    # làm tròn cho gọn prompt
    return {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in ev.items()}


def format_evidence(ev):
    """In bằng chứng thành các dòng 'tên = giá trị   # mô tả' để đưa vào prompt."""
    lines = []
    for k, v in ev.items():
        lines.append(f"{k} = {v}   # {DESCRIPTIONS[k]}")
    return "\n".join(lines)


def check_evidence_no_leakage(df, origin, temp_level="L1", temp_fc=None):
    """Phá phụ tải và nhiệt độ SAU origin; bằng chứng ở L1 không được đổi
    (ở L2/L3 chỉ temp_mean_D_F được phép đổi khi phá nhiệt độ)."""
    base = evidence(df, origin, temp_level, temp_fc)
    res = {}
    for what in ("load", "T"):
        bad = df.copy()
        fut = bad.index >= pd.Timestamp(origin)
        bad.loc[fut, what] = bad.loc[fut, what] * 1.5 + 20
        new = evidence(bad, origin, temp_level, temp_fc)
        changed = [k for k in base if base[k] != new[k]]
        allowed = {"temp_mean_D_F"} if (what == "T" and temp_level == "L3") else set()
        res[what] = {"changed": changed, "ok": set(changed) <= allowed}
    return res
