"""
src/evaluation.py  –  Giai đoạn 3: giao thức đánh giá dùng CHUNG cho mọi phương pháp
(baseline, bộ chọn thống kê, random search, ToT). Cập nhật theo nhận xét của thầy (01/11/2026):

  * 3.1  information_set() có tham số temp_level:
           L1 = chỉ thấy nhiệt độ đến 23:00 ngày D-1        (bản chính)
           L2 = thấy nhiệt độ DỰ BÁO cho ngày D              (từ src/temperature.py)
           L3 = thấy nhiệt độ THẬT của ngày D               (oracle experiment)
  * 3.5  ba cách chọn giai đoạn để chấm điểm: S1, S2, S3
  * 3.6  check_no_leakage() phá hỏng cả phụ tải lẫn nhiệt độ tương lai
  * 3.8  representative_origins(): các origin đại diện cho ToT

Bài toán: tại origin = 00:00 ngày D, dự báo 24 giờ 00:00..23:00 của ngày D.
"""
import copy

import numpy as np
import pandas as pd
from scipy import stats

from src.data import SELECTION_END

HORIZON = 24
TEMP_LEVELS = ("L1", "L2", "L3")

DATA_START = pd.Timestamp("2004-01-01")
SCORE_START = pd.Timestamp("2005-07-01")   # origin sớm nhất được dùng để chấm điểm (cần >= 18 tháng lịch sử)
VAL_START, VAL_END = pd.Timestamp("2006-07-01"), pd.Timestamp("2007-06-30")
TEST_START, TEST_END = pd.Timestamp("2007-07-01"), pd.Timestamp("2008-06-29")

# Các cột mô hình được phép thấy (bỏ T_s1..T_s11, T_mean để tránh lọt nhiệt độ tương lai qua cột khác)
MODEL_COLS = ["load", "is_imputed", "T", "hour", "dayofweek", "month", "dayofyear", "is_holiday"]


# ----------------------------------------------------------------------------
# 3.1 Định nghĩa bài toán
# ----------------------------------------------------------------------------
def daily_origins(start, end):
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")


def period_of(origin):
    if origin < VAL_START:
        return "history"
    if origin <= VAL_END:
        return "validation"
    return "test"


def target_index(origin):
    return pd.date_range(origin, periods=HORIZON, freq="h")


def information_set(df, origin, temp_level="L1", temp_fc=None, lookback_days=None):
    """
    Bảng mô hình được phép nhìn khi dự báo tại `origin`.
      - Phụ tải từ origin trở đi bị xóa (NaN).
      - Biến lịch giữ nguyên (biết trước).
      - Nhiệt độ của 24 giờ mục tiêu tùy temp_level (xem đầu file).
    Bảng gồm cả 24 dòng mục tiêu để mô hình điền dự báo vào.
    """
    assert temp_level in TEMP_LEVELS
    start = DATA_START if lookback_days is None else origin - pd.Timedelta(days=lookback_days)
    end = origin + pd.Timedelta(hours=HORIZON - 1)
    info = df.loc[start:end, MODEL_COLS].copy()
    future = info.index >= origin
    info.loc[future, "load"] = np.nan
    if temp_level == "L1":
        info.loc[future, "T"] = np.nan
    elif temp_level == "L2":
        if temp_fc is None:
            raise ValueError("L2 cần bảng nhiệt độ dự báo temp_fc")
        info.loc[future, "T"] = temp_fc.loc[target_index(origin), "T_fc"].values
    return info


def training_frame(df, origin):
    """Dữ liệu dùng để FIT: mọi giờ trước origin (toàn bộ là quá khứ đã quan sát)."""
    return df.loc[:origin - pd.Timedelta(hours=1), MODEL_COLS].copy()


def actuals(df, origin):
    return df.loc[target_index(origin), "load"]


# ----------------------------------------------------------------------------
# 3.3 Backtest rolling-origin
# ----------------------------------------------------------------------------
def backtest(model, df, origins, temp_level="L1", temp_fc=None, refit="monthly", verbose=False):
    """
    Chạy mô hình qua danh sách origin. Fit lại khi sang tháng mới (refit="monthly")
    bằng toàn bộ dữ liệu trước origin đầu tiên của tháng đó.
    Trả về bảng dài: origin, datetime, y, yhat, is_imputed.
    """
    out, last_key = [], None
    for i, o in enumerate(origins):
        key = (o.year, o.month) if refit == "monthly" else o
        if key != last_key:
            model.fit(training_frame(df, o))
            last_key = key
        info = information_set(df, o, temp_level, temp_fc, lookback_days=model.lookback_days)
        yhat = np.asarray(model.predict(info, o), dtype=float)
        idx = target_index(o)
        out.append(pd.DataFrame({"origin": o, "datetime": idx, "y": df.loc[idx, "load"].values,
                                 "yhat": yhat, "is_imputed": df.loc[idx, "is_imputed"].values}))
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(origins)} origin")
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# 3.4 Metric
# ----------------------------------------------------------------------------
def mase_scale(df):
    """Mẫu số MASE cố định: MAE của seasonal naive lag 24 trên dữ liệu lịch sử (<= SELECTION_END)."""
    s = df.loc[:SELECTION_END, "load"]
    return float((s - s.shift(24)).abs().dropna().mean())


def metrics(fc, scale=None):
    """MAE, RMSE, MAPE, WAPE (= tổng |sai số| / tổng phụ tải, dùng cho zone có giờ phụ tải ≈ 0), MASE."""
    f = fc[~fc["is_imputed"]]
    e = f["y"] - f["yhat"]
    res = {"MAE": e.abs().mean(),
           "RMSE": np.sqrt((e ** 2).mean()),
           "MAPE": (e.abs() / f["y"]).mean() * 100,
           "WAPE": e.abs().sum() / f["y"].abs().sum() * 100}
    if scale is not None:
        res["MASE"] = res["MAE"] / scale
    return res


def daily_loss(fc, kind="mape"):
    """Sai số của từng origin (1 con số / ngày) – dùng cho chấm điểm và kiểm định DM."""
    f = fc[~fc["is_imputed"]]
    e = f["y"] - f["yhat"]
    if kind == "mape":
        v = e.abs() / f["y"] * 100
    elif kind == "wape":
        # WAPE theo ngày = tổng |sai số| / tổng phụ tải của ngày đó (không bị nổ khi có giờ phụ tải ≈ 0)
        g = f["origin"]
        return e.abs().groupby(g).sum() / f["y"].abs().groupby(g).sum() * 100
    elif kind == "se":
        v = e ** 2
    else:
        v = e.abs()
    return v.groupby(f["origin"]).mean()


# ----------------------------------------------------------------------------
# 3.5 Ba cách chọn giai đoạn để chấm điểm tại điểm quyết định d
# ----------------------------------------------------------------------------
def context_features(df):
    """
    Vector ngữ cảnh của mỗi ngày D, CHỈ dùng thông tin đến 23:00 ngày D-1:
      T_prev1    : nhiệt độ trung bình ngày D-1
      Tmax_prev1 : nhiệt độ cao nhất ngày D-1
      T_prev7    : nhiệt độ trung bình 7 ngày trước
      lvl_prev7  : phụ tải TB 7 ngày trước / phụ tải TB 365 ngày trước (mức tải tương đối)
      sin/cos    : vị trí trong năm (mùa)
    """
    d = df[["load", "T"]].resample("D").agg({"load": "mean", "T": ["mean", "max"]})
    d.columns = ["load", "T", "Tmax"]
    feat = pd.DataFrame(index=d.index)
    feat["T_prev1"] = d["T"].shift(1)
    feat["Tmax_prev1"] = d["Tmax"].shift(1)
    feat["T_prev7"] = d["T"].shift(1).rolling(7).mean()
    feat["lvl_prev7"] = d["load"].shift(1).rolling(7).mean() / d["load"].shift(1).rolling(365, min_periods=200).mean()
    doy = feat.index.dayofyear
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return feat


def scoring_origins(d, method, ctx=None, k=21, k_min=10, radius=2.0, return_info=False):
    """
    Tập origin (trước d) dùng để chấm một cấu hình tại điểm quyết định d.
      S1: toàn bộ 365 ngày trước d.
      S2: cùng mùa (ngày 12–18 của tháng d trong 1–3 năm trước) + 7 ngày gần nhất.
      S3: k ngày có ngữ cảnh giống ngày d nhất (cần ctx = context_features(df));
          nếu có ít hơn k_min ngày nằm trong bán kính `radius` -> quay về S2 (fallback).
    """
    d = pd.Timestamp(d)
    if method == "S1":
        out = daily_origins(max(d - pd.Timedelta(days=365), SCORE_START), d - pd.Timedelta(days=1))
        info = "S1"
    elif method == "S2":
        days = []
        for lag in (1, 2, 3):
            y = d.year - lag
            for dd in range(12, 19):
                t = pd.Timestamp(year=y, month=d.month, day=dd)
                if SCORE_START <= t < d:
                    days.append(t)
        days += list(daily_origins(d - pd.Timedelta(days=7), d - pd.Timedelta(days=1)))
        out = pd.DatetimeIndex(sorted(set(days)))
        info = "S2"
    elif method == "S3":
        cand = ctx.loc[SCORE_START:d - pd.Timedelta(days=1)].dropna()
        mu, sd = cand.mean(), cand.std().replace(0, 1)
        z = (cand - mu) / sd
        z0 = (ctx.loc[d] - mu) / sd
        dist = np.sqrt(((z - z0) ** 2).sum(axis=1)).sort_values()
        if (dist <= radius).sum() < k_min:
            out, info = scoring_origins(d, "S2"), "S3->S2 (fallback)"
        else:
            out, info = pd.DatetimeIndex(sorted(dist.index[:k])), "S3"
    else:
        raise ValueError(method)
    return (out, info) if return_info else out


def decision_dates(start, end):
    """Điểm quyết định = ngày 1 mỗi tháng."""
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")


# ----------------------------------------------------------------------------
# 3.6 Kiểm tra rò rỉ
# ----------------------------------------------------------------------------
def check_no_leakage(model, df, origin, temp_level="L1", temp_fc=None):
    """
    Phá hỏng dữ liệu SAU origin rồi dự báo lại:
      - phá phụ tải tương lai  -> dự báo KHÔNG được đổi ở mọi mức
      - phá nhiệt độ tương lai -> L1, L2 không được đổi; L3 được phép đổi (vì L3 dùng nhiệt độ thật)
    """
    def run(frame):
        m = copy.deepcopy(model)
        m.fit(training_frame(frame, origin))
        info = information_set(frame, origin, temp_level, temp_fc, lookback_days=m.lookback_days)
        return np.asarray(m.predict(info, origin), dtype=float)

    base = run(df)
    res = {}
    for what in ("load", "T"):
        bad = df.copy()
        fut = bad.index >= origin
        bad.loc[fut, what] = bad.loc[fut, what] * 1.5 + 20
        diff = float(np.max(np.abs(run(bad) - base)))
        allowed = (what == "T" and temp_level == "L3")
        res[what] = {"max_change": diff, "ok": (diff < 1e-6) or allowed}
    return res


# ----------------------------------------------------------------------------
# 3.7 Kiểm định Diebold–Mariano (HAC Newey–West + hiệu chỉnh HLN)
# ----------------------------------------------------------------------------
def dm_test(loss_a, loss_b, lag=None):
    """H0: hai phương pháp có sai số kỳ vọng bằng nhau. stat < 0 nghĩa là A tốt hơn B."""
    a, b = loss_a.align(loss_b, join="inner")
    d = (a - b).values
    T = len(d)
    if lag is None:
        lag = int(np.floor(T ** (1 / 3)))
    dc = d - d.mean()
    gamma0 = np.mean(dc ** 2)
    var = gamma0 + 2 * sum((1 - k / (lag + 1)) * np.mean(dc[k:] * dc[:-k]) for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(var / T)
    stat *= np.sqrt((T - 1) / T)            # hiệu chỉnh HLN với h = 1 (sai số tính theo ngày)
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return {"mean_diff": d.mean(), "DM": stat, "p_value": p, "n_days": T}


def holm(pvals):
    """Hiệu chỉnh Holm–Bonferroni cho nhiều phép so sánh."""
    p = pd.Series(pvals).sort_values()
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.values)])
    return pd.Series(np.minimum(adj, 1), index=p.index).reindex(pd.Series(pvals).index)


# ----------------------------------------------------------------------------
# 3.8 Origin đại diện cho ToT
# ----------------------------------------------------------------------------
def season_of(month):
    return {12: "dong", 1: "dong", 2: "dong", 3: "xuan", 4: "xuan", 5: "xuan",
            6: "he", 7: "he", 8: "he", 9: "thu", 10: "thu", 11: "thu"}[month]


def representative_origins(df, start, end, n_per_stratum=2, seed=42):
    """
    Lấy mẫu phân tầng: mùa × loại ngày (thường / cuối tuần / lễ) × nhiệt độ (cực trị / bình thường).
    "Cực trị" = nhiệt độ TB ngày nằm trong 10% cao nhất hoặc thấp nhất của mùa đó.
    Đây là bước THIẾT KẾ ĐÁNH GIÁ (chọn ngày để chấm), không phải đầu vào mô hình.
    """
    days = df.loc[start:pd.Timestamp(end) + pd.Timedelta(hours=23)]
    d = days.resample("D").agg({"T": "mean", "is_holiday": "max"})
    d = d.loc[pd.Timestamp(start):pd.Timestamp(end)]
    d["season"] = [season_of(m) for m in d.index.month]
    d["daytype"] = np.where(d["is_holiday"] == 1, "le",
                            np.where(d.index.dayofweek >= 5, "cuoi_tuan", "thuong"))
    lo = d.groupby("season")["T"].transform(lambda s: s.quantile(0.1))
    hi = d.groupby("season")["T"].transform(lambda s: s.quantile(0.9))
    d["temp"] = np.where((d["T"] <= lo) | (d["T"] >= hi), "cuc_tri", "binh_thuong")
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in d.groupby(["season", "daytype", "temp"]):
        n = min(n_per_stratum, len(g))
        picks.extend(rng.choice(g.index.values, size=n, replace=False))
    out = d.loc[sorted(picks), ["season", "daytype", "temp", "T"]]
    out.index.name = "origin"
    return out
