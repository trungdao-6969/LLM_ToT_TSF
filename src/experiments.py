"""
src/experiments.py  –  Chạy backtest có lưu cache, để không phải chạy lại mô hình nhiều lần.

Mỗi (chuỗi, mức nhiệt độ, cấu hình) được lưu thành một file:
    results/forecasts/<series>/<temp_level>/<config_id>.parquet
Lần sau gọi lại sẽ đọc file thay vì chạy lại (trừ khi force=True).
"""
import time
from pathlib import Path

import pandas as pd

from src.data import ROOT
from src.evaluation import backtest, daily_origins, SCORE_START, TEST_END
from src.models import build_model, config_id, USES_TEMPERATURE

FC_DIR = ROOT / "results" / "forecasts"


def run_cached(df, cfg, temp_level, series="system", temp_fc=None,
               start=SCORE_START, end=TEST_END, force=False, verbose=True):
    # Mô hình không dùng nhiệt độ cho kết quả giống nhau ở mọi mức -> chỉ chạy 1 lần, lưu ở L1
    level_used = temp_level if USES_TEMPERATURE[cfg["model"]] else "L1"
    path = FC_DIR / str(series) / level_used / f"{config_id(cfg)}.parquet"
    origins = daily_origins(start, end)
    if path.exists() and not force:
        fc = pd.read_parquet(path)
        if set(origins) <= set(fc["origin"].unique()):
            return fc[fc["origin"].isin(origins)].reset_index(drop=True)
    t = time.time()
    fc = backtest(build_model(cfg, level_used), df, origins, level_used, temp_fc)
    path.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(path)
    if verbose:
        print(f"  xong {config_id(cfg)} [{level_used}] trong {time.time() - t:.0f}s")
    return fc
