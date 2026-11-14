"""
src/eda.py  –  Mục 2.14: mô tả đặc trưng từng zone và chọn zone đại diện bằng dữ liệu.

Ba nhóm zone thầy yêu cầu:
  (1) mùa vụ ổn định  -> hình dạng ngày giống nhau qua các ngày (prof_stab cao)
  (2) biến động mạnh  -> phụ tải trung bình ngày thay đổi nhiều giữa hai ngày liên tiếp (vol cao)
  (3) đổi chế độ      -> mức tải năm nay khác hẳn cùng tháng năm trước (max_yoy cao)

Lưu ý về dữ liệu dùng để chọn:
  - (1) và (2) chỉ tính trên dữ liệu đến SELECTION_END.
  - (3) phải nhìn cả giai đoạn thì mới thấy được bước nhảy. Việc này là THIẾT KẾ THÍ NGHIỆM
    (chọn chuỗi nào để kiểm tra phương pháp khi có đổi chế độ), không phải đầu vào của mô hình:
    mọi mô hình vẫn chỉ được thấy dữ liệu trước origin. Cần ghi rõ điều này trong báo cáo.
"""
import numpy as np
import pandas as pd

from src.data import PROCESSED, SELECTION_END, N_ZONES


def zone_matrix():
    z = pd.read_parquet(PROCESSED / "zones_long.parquet")
    return z.pivot(index="datetime", columns="zone_id", values="load")


def profile_stability(s):
    """Tương quan trung bình giữa hình dạng 24 giờ của từng ngày và hình dạng trung bình."""
    n = len(s) // 24 * 24
    prof = s.values[:n].reshape(-1, 24)
    prof = prof / prof.mean(axis=1, keepdims=True)
    avg = prof.mean(axis=0)
    return float(np.mean([np.corrcoef(p, avg)[0, 1] for p in prof]))


def find_duplicates(W):
    """Các cặp zone có dữ liệu giống hệt nhau (đã biết: zone 3 = zone 7)."""
    pairs = []
    for i in W.columns:
        for j in W.columns:
            if i < j and W[i].equals(W[j]):
                pairs.append((i, j))
    return pairs


def zone_features(W=None):
    if W is None:
        W = zone_matrix()
    sel = W[W.index <= SELECTION_END]
    rows = {}
    for c in W.columns:
        s = sel[c]
        daily = s.resample("D").mean()
        monthly_full = W[c].resample("MS").mean()
        yoy = np.log(monthly_full / monthly_full.shift(12))
        rows[c] = {
            "mean_kW": s.mean(),
            "cv": s.std() / s.mean(),
            "acf24": s.autocorr(24),
            "acf168": s.autocorr(168),
            "prof_stab": profile_stability(s),
            "vol": daily.pct_change().abs().mean(),
            "max_yoy": yoy.abs().max(),
            "month_max_yoy": yoy.abs().idxmax(),
        }
    feat = pd.DataFrame(rows).T
    feat.index.name = "zone"
    return feat


def choose_representative(feat, duplicates=((3, 7),)):
    """Chọn 3 zone khác nhau theo 3 tiêu chí; bỏ zone trùng (giữ zone nhỏ hơn của mỗi cặp)."""
    drop = {b for a, b in duplicates}
    f = feat.drop(index=[z for z in drop if z in feat.index])
    chosen = {}
    for name, col in [("on_dinh", "prof_stab"), ("bien_dong", "vol"), ("doi_che_do", "max_yoy")]:
        order = f[col].astype(float).sort_values(ascending=False).index
        pick = next(z for z in order if z not in chosen.values())
        chosen[name] = int(pick)
    return chosen
