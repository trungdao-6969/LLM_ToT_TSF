"""
src/temperature.py  –  Mục 4.0: dự báo nhiệt độ ngày D (24 giờ) để dùng cho mức L2.

GEFCom2012 không có dữ liệu dự báo thời tiết lịch sử, nên phải tự xây một mô hình
dự báo nhiệt độ. Mọi dự báo cho ngày D chỉ dùng nhiệt độ đến 23:00 ngày D-1.

Ba phương án (từ đơn giản đến phức tạp):
  persistence : T(D, h) = T(D-1, h)                                   (nhiệt độ cùng giờ hôm qua)
  climatology : trung bình T cùng giờ, ±7 ngày quanh cùng ngày ở 1–3 năm trước
  regression  : với mỗi giờ h, hồi quy T(D,h) ~ T(D-1,h) + T(D-1,23:00) + climatology(D,h),
                fit lại đầu mỗi tháng bằng các ngày trước tháng đó (cửa sổ mở rộng)
Chọn phương án có MAE nhỏ nhất trên VALIDATION, rồi dùng cho cả validation và test.
"""
import numpy as np
import pandas as pd

from src.evaluation import VAL_START, VAL_END, SCORE_START


def day_by_hour(series):
    """Chuỗi giờ -> bảng (ngày × 24 giờ)."""
    s = series.copy()
    return pd.DataFrame({"T": s.values, "date": s.index.normalize(), "hour": s.index.hour}) \
             .pivot(index="date", columns="hour", values="T")


def forecast_all_methods(T):
    Td = day_by_hour(T)
    persist = Td.shift(1)
    roll = Td.rolling(15, center=True, min_periods=10).mean()
    clim = pd.concat([roll.shift(365 * k) for k in (1, 2, 3)]).groupby(level=0).mean()
    clim = clim.reindex(Td.index)
    last = Td.shift(1)[23]

    # Hồi quy theo từng giờ, fit lại đầu mỗi tháng (chỉ dùng các ngày đã qua)
    reg = pd.DataFrame(np.nan, index=Td.index, columns=Td.columns)
    months = pd.date_range(SCORE_START - pd.DateOffset(months=6), Td.index.max(), freq="MS")
    for m in months:
        train_days = Td.index[(Td.index < m)]
        pred_days = Td.index[(Td.index >= m) & (Td.index < m + pd.DateOffset(months=1))]
        for h in range(24):
            X = pd.DataFrame({"p": persist[h], "last": last, "c": clim[h]})
            Xt = X.loc[train_days].dropna()
            yt = Td.loc[Xt.index, h]
            if len(Xt) < 60:
                continue
            A = np.column_stack([np.ones(len(Xt)), Xt.values])
            beta, *_ = np.linalg.lstsq(A, yt.values, rcond=None)
            Xp = X.loc[pred_days]
            reg.loc[pred_days, h] = np.column_stack([np.ones(len(Xp)), Xp.values]) @ beta
    return {"actual": Td, "persistence": persist, "climatology": clim, "regression": reg}


def evaluate_methods(fcs, start=VAL_START, end=VAL_END):
    act = fcs["actual"].loc[start:end]
    rows = {}
    for name in ("persistence", "climatology", "regression"):
        err = (fcs[name].loc[start:end] - act).abs()
        rows[name] = {"MAE_F": err.values.mean(), "MAE_F_gio_xau_nhat": err.mean().max()}
    return pd.DataFrame(rows).T.sort_values("MAE_F")


def to_hourly(Td):
    s = Td.stack()
    idx = s.index.get_level_values(0) + pd.to_timedelta(s.index.get_level_values(1), unit="h")
    return pd.DataFrame({"T_fc": s.values}, index=pd.DatetimeIndex(idx, name="datetime")).sort_index()


def build_temp_forecast(df):
    """Trả về (temp_fc theo giờ cho phương án tốt nhất, bảng MAE, tên phương án)."""
    fcs = forecast_all_methods(df["T"])
    table = evaluate_methods(fcs)
    best = table.index[0]
    return to_hourly(fcs[best]), table, best, fcs
