"""
src/selectors.py  –  Giai đoạn 5: bộ chọn mô hình THUẦN THỐNG KÊ (không LLM).
Đây là "đối thủ" mà ToT phải thắng (yêu cầu của thầy).

Mọi bộ chọn dùng chung:
  - thực đơn cấu hình SEARCH_SPACE (src/models.py),
  - hàm chấm score(cấu hình, d, S) = MAPE trung bình trên tập origin scoring_origins(d, S),
  - quy tắc: chọn ở ngày 1 mỗi tháng, dùng cấu hình đó cho cả tháng.

Đầu vào chung:
  fcs  : dict config_id -> bảng dự báo (origin, datetime, y, yhat, is_imputed) từ run_cached
  loss : DataFrame (origin × config_id) = MAPE theo ngày
"""
import numpy as np
import pandas as pd

from src.evaluation import daily_loss, scoring_origins, decision_dates


def loss_matrix(fcs, kind="mape"):
    return pd.DataFrame({k: daily_loss(v, kind) for k, v in fcs.items()})


def score(loss, cfg_ids, d, method, ctx=None):
    S = scoring_origins(d, method, ctx)
    return loss.loc[loss.index.isin(S), cfg_ids].mean()


def month_origins(d, fc_index):
    end = d + pd.DateOffset(months=1)
    return fc_index[(fc_index >= d) & (fc_index < end)]


def assemble(fcs, choice):
    """choice: Series (ngày quyết định -> config_id). Ghép dự báo của cấu hình được chọn cho từng tháng."""
    parts = []
    for d, cid in choice.items():
        f = fcs[cid]
        o = f["origin"]
        parts.append(f[(o >= d) & (o < d + pd.DateOffset(months=1))])
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# 5.2 Chọn tĩnh: chọn 1 lần trên toàn bộ validation, dùng cho cả giai đoạn sau
# ----------------------------------------------------------------------------
def static_choice(loss, start, end, cfg_ids=None):
    cfg_ids = cfg_ids or list(loss.columns)
    return loss.loc[start:end, cfg_ids].mean().idxmin()


# ----------------------------------------------------------------------------
# 5.3 Chọn động: mỗi tháng chọn cấu hình có điểm tốt nhất theo S1/S2/S3
# ----------------------------------------------------------------------------
def dynamic_choice(loss, start, end, method, ctx=None, cfg_ids=None):
    cfg_ids = cfg_ids or list(loss.columns)
    return pd.Series({d: score(loss, cfg_ids, d, method, ctx).idxmin()
                      for d in decision_dates(start, end)})


# ----------------------------------------------------------------------------
# 5.4 Ensemble: trung bình đơn giản hoặc trọng số nghịch đảo sai số (tính lại mỗi tháng)
# ----------------------------------------------------------------------------
def ensemble(fcs, loss, pool, start, end, method=None, ctx=None):
    parts = []
    for d in decision_dates(start, end):
        if method is None:
            w = pd.Series(1.0, index=pool)
        else:
            s = score(loss, pool, d, method, ctx)
            w = 1.0 / s
        w = w / w.sum()
        months = [fcs[c][(fcs[c]["origin"] >= d) & (fcs[c]["origin"] < d + pd.DateOffset(months=1))]
                  .reset_index(drop=True) for c in pool]
        out = months[0][["origin", "datetime", "y", "is_imputed"]].copy()
        out["yhat"] = sum(w[c] * m["yhat"].values for c, m in zip(pool, months))
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# 5.5 Random search: mỗi tháng bốc ngẫu nhiên `budget` cấu hình, giữ cấu hình điểm tốt nhất
#     (cùng ngân sách 9 lần backtest với ToT)
# ----------------------------------------------------------------------------
def random_search_choice(loss, start, end, method, ctx=None, budget=9, seed=0):
    rng = np.random.default_rng(seed)
    cols = np.array(loss.columns)
    out = {}
    for d in decision_dates(start, end):
        cand = list(rng.choice(cols, size=budget, replace=False))
        out[d] = score(loss, cand, d, method, ctx).idxmin()
    return pd.Series(out)


# ----------------------------------------------------------------------------
# 5.6 Oracle (cận trên, biết trước đáp án – không dùng được trong thực tế)
# ----------------------------------------------------------------------------
def oracle_monthly_choice(loss, start, end):
    """Mỗi tháng chọn cấu hình tốt nhất CỦA CHÍNH THÁNG ĐÓ (cận trên cho bộ chọn theo tháng)."""
    return pd.Series({d: loss.loc[d:d + pd.DateOffset(months=1) - pd.Timedelta(days=1)].mean().idxmin()
                      for d in decision_dates(start, end)})


def oracle_daily(fcs, loss, start, end):
    """Mỗi NGÀY chọn cấu hình tốt nhất ngày đó (cận trên tuyệt đối)."""
    best = loss.loc[start:end].idxmin(axis=1)
    parts = [fcs[c][fcs[c]["origin"] == o] for o, c in best.items()]
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# Độ tin cậy của cách chấm: điểm (trên tập S) có xếp hạng đúng sai số thật của tháng sau không?
# ----------------------------------------------------------------------------
def score_reliability(loss, start, end, method, ctx=None):
    from scipy.stats import spearmanr
    rows = []
    for d in decision_dates(start, end):
        s = score(loss, list(loss.columns), d, method, ctx)
        real = loss.loc[d:d + pd.DateOffset(months=1) - pd.Timedelta(days=1)].mean()
        rho = spearmanr(s, real).correlation
        regret = real[s.idxmin()] - real.min()     # tháng đó mất bao nhiêu % MAPE so với lựa chọn tốt nhất
        rows.append({"d": d, "spearman": rho, "regret_MAPE": regret})
    return pd.DataFrame(rows).set_index("d")
