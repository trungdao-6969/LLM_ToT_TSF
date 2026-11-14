"""
src/models.py  –  Giai đoạn 4: các mô hình baseline + "thực đơn" cấu hình (registry).

Mỗi mô hình có 2 hàm:
    fit(train)            : train = mọi giờ trước origin (gọi đầu mỗi tháng)
    predict(info, origin) : info = information_set(...) -> trả về 24 giá trị dự báo

Cách xử lý nhiệt độ theo mức (temp_level):
    L1 -> mô hình chỉ dùng nhiệt độ TRỄ 24 giờ (T của cùng giờ ngày D-1) và thống kê ngày D-1
    L2 -> dùng T của giờ mục tiêu = nhiệt độ DỰ BÁO (khi fit vẫn dùng nhiệt độ thật trong quá khứ)
    L3 -> dùng T của giờ mục tiêu = nhiệt độ THẬT (oracle)
Ở L1, information_set đã xóa T tương lai, nên nếu mô hình lỡ dùng T(t) thì sẽ ra NaN -> lỗi ngay.

Một cấu hình là một dict, ví dụ {"model": "mlr", "train_years": 2, "poly": 3, "month_temp": True}.
LLM (ở GĐ6) và random search (ở GĐ5) chỉ được chọn cấu hình trong SEARCH_SPACE.
"""
import itertools
import warnings

import numpy as np
import pandas as pd

from src.evaluation import DATA_START, HORIZON, target_index


# ----------------------------------------------------------------------------
# Hàm phụ
# ----------------------------------------------------------------------------
def temp_column(frame, temp_level):
    """Nhiệt độ dùng làm đặc trưng: L1 -> trễ 24 giờ; L2/L3 -> nhiệt độ của chính giờ đó."""
    return frame["T"].shift(24) if temp_level == "L1" else frame["T"]


def recent_years(train, years):
    if years is None:
        return train
    return train[train.index >= train.index.max() - pd.DateOffset(years=years)]


def daily_stat_prev(frame, col, how="mean"):
    """Thống kê của NGÀY HÔM TRƯỚC (D-1), gắn vào mọi giờ của ngày D."""
    day = frame.index.normalize()
    stat = frame[col].groupby(day).agg(how)
    prev = stat.shift(1, freq="D")
    return pd.Series(prev.reindex(day).values, index=frame.index)


# ----------------------------------------------------------------------------
# 1. Seasonal Naive
# ----------------------------------------------------------------------------
class SeasonalNaive:
    def __init__(self, lag=24, temp_level="L1"):
        self.lag = lag
        self.lookback_days = 8

    def fit(self, train):
        pass

    def predict(self, info, origin):
        idx = target_index(origin) - pd.Timedelta(hours=self.lag)
        return info.loc[idx, "load"].values


# ----------------------------------------------------------------------------
# 2. MLR của Tao Hong ("vanilla benchmark")
#    Load = b0 + b1*Trend + M + W*H + H*(T,T²,T³) + M*(T,T²,T³)
# ----------------------------------------------------------------------------
class MLRTaoHong:
    def __init__(self, train_years=None, poly=3, month_temp=True, temp_level="L1"):
        self.train_years = train_years
        self.poly = poly
        self.month_temp = month_temp
        self.temp_level = temp_level
        self.lookback_days = 3
        self.beta = None

    def design(self, frame):
        n = len(frame)
        hour = frame["hour"].values
        dow = frame["dayofweek"].values
        month = frame["month"].values
        trend = ((frame.index - DATA_START) / pd.Timedelta(hours=1)).values / 10000.0
        T = (temp_column(frame, self.temp_level).values - 60.0) / 20.0   # chuẩn hóa cho ổn định số

        cols = [np.ones(n), trend]
        M = np.eye(12)[month - 1]
        WH = np.eye(168)[dow * 24 + hour]
        H = np.eye(24)[hour]
        cols += [M[:, 1:], WH[:, 1:]]
        for p in range(1, self.poly + 1):
            cols.append(H * (T ** p)[:, None])
            if self.month_temp:
                cols.append(M * (T ** p)[:, None])
        return np.column_stack(cols)

    def fit(self, train):
        tr = recent_years(train, self.train_years)
        X = self.design(tr)
        y = tr["load"].values
        ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        self.beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)

    def predict(self, info, origin):
        X = self.design(info)[-HORIZON:]
        return X @ self.beta


# ----------------------------------------------------------------------------
# 3. MSTL + ETS: tách mùa vụ ngày (24) và tuần (168), ETS (Holt damped) cho phần còn lại,
#    mùa vụ được dự báo bằng cách lặp lại chu kỳ gần nhất. Không dùng nhiệt độ.
# ----------------------------------------------------------------------------
class MSTLETS:
    def __init__(self, window_weeks=8, temp_level="L1"):
        self.window_weeks = window_weeks
        self.lookback_days = window_weeks * 7 + 1

    def fit(self, train):
        pass   # fit lại ở mỗi origin vì cần trạng thái mới nhất

    def predict(self, info, origin):
        from statsmodels.tsa.seasonal import MSTL
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y = info.loc[:origin - pd.Timedelta(hours=1), "load"].iloc[-self.window_weeks * 168:]
        res = MSTL(y.values, periods=(24, 168)).fit()
        seas = res.seasonal                         # cột 0: chu kỳ 24, cột 1: chu kỳ 168
        s24 = seas[-24:, 0]
        s168 = seas[-168:-168 + 24, 1]
        rest = res.trend + res.resid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ets = ExponentialSmoothing(rest, trend="add", damped_trend=True).fit()
        return ets.forecast(HORIZON) + s24 + s168


# ----------------------------------------------------------------------------
# 4. SARIMAX(2,0,0)(1,1,0)_24 + Fourier chu kỳ tuần + nhiệt độ (T, T²)
#    Ước lượng tham số đầu mỗi tháng trên `window_weeks` tuần gần nhất;
#    các ngày trong tháng chỉ "lọc" lại dữ liệu mới bằng tham số đã có (apply), không ước lượng lại.
# ----------------------------------------------------------------------------
class SARIMAXModel:
    def __init__(self, window_weeks=4, temp_level="L1"):
        self.window_weeks = window_weeks
        self.temp_level = temp_level
        self.lookback_days = window_weeks * 7 + 2
        self.res = None

    def exog(self, frame):
        T = (temp_column(frame, self.temp_level) - 60.0) / 20.0
        how = (frame["dayofweek"] * 24 + frame["hour"]).values
        cols = {"T": T, "T2": T ** 2}
        for k in (1, 2):
            cols[f"sin{k}"] = np.sin(2 * np.pi * k * how / 168)
            cols[f"cos{k}"] = np.cos(2 * np.pi * k * how / 168)
        return pd.DataFrame(cols, index=frame.index)

    def _window(self, frame, end):
        w = frame.loc[:end].iloc[-self.window_weeks * 168:]
        return w["load"].values / self.scale, self.exog(frame).loc[w.index].values

    def fit(self, train):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.scale = train["load"].iloc[-self.window_weeks * 168:].mean()
        y, X = self._window(train, train.index.max())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(y, exog=X, order=(2, 0, 0), seasonal_order=(1, 1, 0, 24))
            self.res = mod.fit(disp=False, maxiter=50)

    def predict(self, info, origin):
        y, X = self._window(info, origin - pd.Timedelta(hours=1))
        Xf = self.exog(info).loc[target_index(origin)].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = self.res.apply(y, exog=X, refit=False)
            return r.forecast(HORIZON, exog=Xf) * self.scale


# ----------------------------------------------------------------------------
# 5. LightGBM (một mô hình cho cả 24 giờ, đặc trưng đều là trễ >= 24 giờ)
#    target="raw"  : học trực tiếp phụ tải
#    target="ratio": học tỉ số phụ tải / phụ tải TB ngày D-1 (giúp thích nghi khi mức tải đổi)
# ----------------------------------------------------------------------------
class LightGBMModel:
    def __init__(self, num_leaves=31, target="ratio", train_years=None, n_estimators=400,
                 learning_rate=0.05, temp_level="L1", seed=0):
        self.params = dict(num_leaves=num_leaves, n_estimators=n_estimators,
                           learning_rate=learning_rate, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, verbose=-1, n_jobs=2)
        self.target = target
        self.train_years = train_years
        self.temp_level = temp_level
        self.lookback_days = 20

    def features(self, frame):
        F = frame[["hour", "dayofweek", "month", "dayofyear", "is_holiday"]].copy()
        for k in (24, 48, 72, 168, 336):
            F[f"lag{k}"] = frame["load"].shift(k)
        F["load_mean_D1"] = daily_stat_prev(frame, "load", "mean")
        F["load_max_D1"] = daily_stat_prev(frame, "load", "max")
        F["T_lag24"] = frame["T"].shift(24)
        F["T_mean_D1"] = daily_stat_prev(frame, "T", "mean")
        F["T_max_D1"] = daily_stat_prev(frame, "T", "max")
        if self.temp_level != "L1":
            F["T"] = frame["T"]
            F["T2"] = frame["T"] ** 2
            F["T_mean_D"] = frame["T"].groupby(frame.index.normalize()).transform("mean")
        if self.target == "ratio":
            for c in [c for c in F.columns if c.startswith("lag") or c.startswith("load_")]:
                F[c] = F[c] / F["load_mean_D1"]
            F["level"] = F["load_mean_D1"]
            F = F.drop(columns="load_mean_D1")
        return F

    def fit(self, train):
        import lightgbm as lgb
        tr = recent_years(train, self.train_years)
        F = self.features(tr)
        y = tr["load"]
        if self.target == "ratio":
            y = y / daily_stat_prev(tr, "load", "mean")
        ok = F.notna().all(axis=1) & y.notna()
        self.model = lgb.LGBMRegressor(**self.params).fit(F[ok], y[ok])

    def predict(self, info, origin):
        F = self.features(info).loc[target_index(origin)]
        p = self.model.predict(F)
        if self.target == "ratio":
            p = p * daily_stat_prev(info, "load", "mean").loc[target_index(origin)].values
        return p


# ----------------------------------------------------------------------------
# Registry: tên mô hình -> lớp; và "thực đơn" cấu hình
# ----------------------------------------------------------------------------
REGISTRY = {"snaive": SeasonalNaive, "mlr": MLRTaoHong, "mstl_ets": MSTLETS,
            "sarimax": SARIMAXModel, "lgbm": LightGBMModel}

USES_TEMPERATURE = {"snaive": False, "mlr": True, "mstl_ets": False, "sarimax": True, "lgbm": True}


def build_model(cfg, temp_level="L1"):
    params = {k: v for k, v in cfg.items() if k != "model"}
    return REGISTRY[cfg["model"]](temp_level=temp_level, **params)


def config_id(cfg):
    """Tên ngắn, dùng làm tên file cache. vd mlr_train_years-2_poly-3_month_temp-True"""
    parts = [cfg["model"]] + [f"{k}-{v}" for k, v in sorted(cfg.items()) if k != "model"]
    return "_".join(parts)


# Baseline mặc định cho bảng GĐ4
BASELINES = {
    "snaive24": {"model": "snaive", "lag": 24},
    "snaive168": {"model": "snaive", "lag": 168},
    "mlr": {"model": "mlr", "train_years": None, "poly": 3, "month_temp": True},
    "mstl_ets": {"model": "mstl_ets", "window_weeks": 8},
    "sarimax": {"model": "sarimax", "window_weeks": 4},
    "lgbm": {"model": "lgbm", "num_leaves": 31, "target": "ratio", "train_years": None},
}

# Không gian tìm kiếm cho GĐ5 (random search, bộ chọn) và GĐ6 (LLM chỉ được chọn trong này).
# SARIMAX chỉ là baseline: ước lượng chậm nên không đưa vào không gian tìm kiếm.
SEARCH_SPACE = {
    "snaive": {"lag": [24, 168]},
    "mlr": {"train_years": [1, 2, None], "poly": [2, 3], "month_temp": [True, False]},
    "mstl_ets": {"window_weeks": [4, 8]},
    "lgbm": {"num_leaves": [31, 63], "target": ["raw", "ratio"], "train_years": [2, None]},
}


def all_configs(space=SEARCH_SPACE):
    out = []
    for m, grid in space.items():
        keys = list(grid)
        for vals in itertools.product(*[grid[k] for k in keys]):
            out.append({"model": m, **dict(zip(keys, vals))})
    return out
