"""
src/tot.py  –  GĐ6: Tree of Thoughts chọn cấu hình dự báo cho từng origin.

Bước 6.4 (phần 1): Scorer = "evaluator" của ToT. Chấm một cấu hình tại origin d bằng DỮ LIỆU:
    score(cfg, d) = sai số trung bình theo ngày của cfg trên tập origin chấm điểm trước d (S1 mặc định)
  - cấu hình chưa có dự báo -> backtest MỘT lần trên cả giai đoạn SCORE_START..TEST_END rồi lưu cache
    (run_cached), những lần sau chỉ đọc lại;
  - không rò rỉ: dự báo tại origin o chỉ dùng thông tin trước o, và score(d) chỉ đọc các origin < d.
    Dự báo tại chính ngày d chỉ dùng để ĐÁNH GIÁ sau khi đã chọn xong (loss_at).
Bước 6.5 (phần 2): bộ điều khiển ToT cho một origin (k = 3, b = 2, depth 2) + cửa sổ chấm điểm.
"""
import json
import math
import time

import pandas as pd

from src.data import ROOT
from src.evaluation import daily_loss, daily_origins, scoring_origins
from src.evidence import evidence, format_evidence
from src.experiments import run_cached
from src.models import config_id
from src.thoughts import generate_tier1, generate_tier2

LOG_DIR = ROOT / "results" / "tot_logs"


class Scorer:
    def __init__(self, df, series, temp_level="L1", temp_fc=None, method="S1", ctx=None):
        self.df, self.series, self.temp_level, self.temp_fc = df, series, temp_level, temp_fc
        self.kind = "mape" if series == "system" else "wape"   # zone: WAPE (có giờ tải ≈ 0)
        self.method, self.ctx = method, ctx
        self._loss = {}                                       # config_id -> loss theo ngày
        self.new_backtests = []                               # (config_id, giây) cấu hình phải chạy mới

    def daily(self, cfg):
        """Sai số theo ngày (Series: origin -> loss) của cấu hình trên cả giai đoạn."""
        cid = config_id(cfg)
        if cid not in self._loss:
            t0 = time.time()
            fc = run_cached(self.df, cfg, self.temp_level, series=self.series, temp_fc=self.temp_fc,
                            verbose=False)
            dt = time.time() - t0
            if dt > 2:                                         # đọc file cache chỉ mất < 1 giây
                self.new_backtests.append((cid, round(dt, 1)))
            self._loss[cid] = daily_loss(fc, self.kind)
        return self._loss[cid]

    def score(self, cfg, d, window="365d"):
        """Điểm tại origin d: trung bình sai số trên các ngày TRƯỚC d.
        window "365d" -> scoring_origins(d, method) (S1 mặc định); "28d"/"7d" -> 28/7 ngày gần nhất."""
        d = pd.Timestamp(d)
        if window == "365d":
            S = scoring_origins(d, self.method, self.ctx)
        else:
            n = int(window[:-1])
            S = daily_origins(d - pd.Timedelta(days=n), d - pd.Timedelta(days=1))
        loss = self.daily(cfg)
        return float(loss[loss.index.isin(S)].mean())

    def loss_at(self, cfg, d):
        """Sai số THẬT của cấu hình tại ngày d (dùng để đánh giá sau khi đã chọn)."""
        return float(self.daily(cfg).get(pd.Timestamp(d), float("nan")))


# ============================================================================
# Bước 6.5: bộ điều khiển ToT cho MỘT origin
# ============================================================================
SIMPLICITY = {"snaive": 0, "mstl_ets": 1, "mlr": 2, "lgbm": 3}   # hòa điểm -> chọn mô hình đơn giản hơn


def _best(nodes, key):
    ok = [n for n in nodes if n["valid"] and n.get(key) is not None and not math.isnan(n[key])]
    return min(ok, key=lambda n: (n[key], SIMPLICITY[n["config"]["model"]])) if ok else None


def run_tot(scorer, d, k=3, b=2, scoring_mode="llm", temp_fc=None, tag="", temperature=0.7, save=True):
    """Chạy ToT tại origin d:
      tầng 1: LLM đề xuất k thought (+ chọn cửa sổ chấm nếu scoring_mode="llm") -> chấm -> giữ b tốt nhất
      tầng 2: mỗi thought được giữ -> LLM tinh chỉnh k con -> chấm
      chọn  : node có điểm tốt nhất trong cả b*k + k node.
    Trả về bản ghi (dict) và lưu vào results/tot_logs/<series>/<level>/<mode>/<d>.json"""
    d = pd.Timestamp(d)
    t0 = time.time()
    metric = scorer.kind.upper()
    prefix = f"{scorer.series}_{d.date()}"
    ev = evidence(scorer.df, d, scorer.temp_level, temp_fc if temp_fc is not None else scorer.temp_fc)
    ev_text = format_evidence(ev)

    # ---- tầng 1
    tier1, log1, window = generate_tier1(ev_text, scorer.temp_level, prefix, metric, k, tag, temperature,
                                         ask_window=(scoring_mode == "llm"))
    for t in tier1:
        t["score"] = scorer.score(t["config"], d, window) if t["valid"] else None
    ranked = sorted([t for t in tier1 if t["valid"]],
                    key=lambda t: (t["score"], SIMPLICITY[t["config"]["model"]]))
    parents = ranked[:b]

    # ---- tầng 2
    tier2, logs2 = [], []
    for p in parents:
        kids, log2 = generate_tier2(ev_text, scorer.temp_level, tier1, p, prefix, metric, k, tag,
                                    temperature, window)
        for c in kids:
            c["score"] = scorer.score(c["config"], d, window) if c["valid"] else None
        tier2 += kids
        logs2.append(log2)

    # ---- chọn cuối + đánh giá (sai số thật tại d chỉ đọc SAU khi đã chọn)
    nodes = tier1 + tier2
    for n in nodes:
        if n["valid"]:
            n["score_365d"] = scorer.score(n["config"], d, "365d")   # để so sánh: nếu chỉ chấm bằng S1
            n["loss_at_d"] = scorer.loss_at(n["config"], d)
    chosen = _best(nodes, "score")
    chosen_s1 = _best(nodes, "score_365d")
    best_in_tree = _best(nodes, "loss_at_d")                     # oracle trong 9 node (cận trên của cây)
    distinct = {config_id(n["config"]) for n in nodes if n["valid"]}

    rec = {
        "series": scorer.series, "origin": str(d.date()), "temp_level": scorer.temp_level,
        "scoring_mode": scoring_mode, "window": window, "metric": metric, "tag": tag,
        "chosen": chosen["config"] if chosen else None,
        "chosen_id": chosen["thought_id"] if chosen else None,
        "loss_chosen": chosen["loss_at_d"] if chosen else None,
        "loss_chosen_if_365d": chosen_s1["loss_at_d"] if chosen_s1 else None,
        "loss_best_in_tree": best_in_tree["loss_at_d"] if best_in_tree else None,
        "n_valid": sum(n["valid"] for n in nodes), "n_distinct_configs": len(distinct),
        "n_issues": sum(len(n["issues"]) for n in nodes),
        "seconds": round(time.time() - t0, 1),
        "evidence": ev, "nodes": nodes,
        "llm_logs": {"tier1": log1, "tier2": logs2, "window_issue": log1.get("window_issue")},
    }
    if save:
        path = LOG_DIR / str(scorer.series) / scorer.temp_level / scoring_mode / f"{d.date()}{tag}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return rec


def print_record(rec):
    print(f"[{rec['series']} {rec['origin']} {rec['temp_level']} {rec['scoring_mode']}] "
          f"cửa sổ chấm = {rec['window']}, {rec['n_distinct_configs']} cấu hình khác nhau, "
          f"{rec['seconds']:.0f}s")
    print(f"  {'node':28s} {'category':16s} {'score':>7s} {'365d':>7s} {'thật@d':>7s}  config")
    for n in rec["nodes"]:
        if not n["valid"]:
            print(f"  {n['thought_id']:28s} KHÔNG HỢP LỆ: {n['issues']}")
            continue
        mark = " <== chọn" if n["thought_id"] == rec["chosen_id"] else ""
        cfg = {k: v for k, v in n["config"].items()}
        print(f"  {n['thought_id']:28s} {n['hypothesis']['category']:16s} {n['score']:7.2f} "
              f"{n['score_365d']:7.2f} {n['loss_at_d']:7.2f}  {cfg}{mark}")
    print(f"  => sai số ToT: {rec['loss_chosen']:.2f} | nếu chấm bằng 365d: {rec['loss_chosen_if_365d']:.2f}"
          f" | tốt nhất trong cây: {rec['loss_best_in_tree']:.2f}")


def _demo_run():
    from src.data import load_dataset
    from src.llm import settings, STATS
    print("LLM:", settings()["backend"], settings()["model"])
    sc = Scorer(load_dataset(10), series=10, temp_level="L1")
    rec = run_tot(sc, "2008-01-10")
    print_record(rec)
    print("LLM:", STATS, "| backtest mới:", sc.new_backtests)


# ----------------------------------------------------------------------------
# Chạy thử: python -m src.tot score   (bước 6.4, không gọi LLM)
#           python -m src.tot run     (bước 6.5, ToT đầy đủ cho zone 10 ngày 10/01/2008, gọi LLM 3 lần)
#   score: chấm 3 thought mà llama31-8k đã đề xuất cho zone 10 ngày 10/01/2008 (bước 6.3)
# ----------------------------------------------------------------------------
def _demo_score():
    from src.data import load_dataset
    d = "2008-01-10"
    sc = Scorer(load_dataset(10), series=10, temp_level="L1")
    thoughts = {
        "t1 stable_seasonal": {"model": "mstl_ets", "window_weeks": 8},
        "t2 level_shift": {"model": "mlr", "train_years": 2, "poly": 3, "month_temp": True,
                           "holiday": False, "recent_bias": True},
        "t3 temp_sensitive": {"model": "mlr", "train_years": 2, "poly": 2, "month_temp": True,
                              "holiday": False, "recent_bias": False},
        "(so sánh) lgbm ratio 1 năm": {"model": "lgbm", "num_leaves": 31, "target": "ratio",
                                       "train_years": 1, "n_estimators": 400, "lags": "short"},
    }
    print(f"zone 10, origin {d}, chấm bằng {sc.method} ({sc.kind.upper()} %)")
    print(f"{'thought':30s} {'score (365 ngày trước)':>24s} {'sai số thật ngày d':>20s}")
    for name, cfg in thoughts.items():
        print(f"{name:30s} {sc.score(cfg, d):24.2f} {sc.loss_at(cfg, d):20.2f}")
    print("Cấu hình phải backtest mới:", sc.new_backtests)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        _demo_score()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        _demo_run()
