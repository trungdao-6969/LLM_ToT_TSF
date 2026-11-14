"""
src/data.py  –  Giai đoạn 2: xử lý dữ liệu GEFCom2012 (track Load)

Đầu vào : các file CSV gốc trong data/raw/
Đầu ra  : data/processed/zones_long.parquet   (bảng dài: datetime × zone)
          data/processed/temperature.parquet  (nhiệt độ 11 trạm, theo giờ)
          data/processed/calendar.parquet     (biến lịch + ngày lễ)
          data/processed/zone_station.json    (trạm nhiệt độ gán cho từng zone, mục 2.15)

Các notebook chỉ cần gọi load_dataset(series) để lấy một bảng giờ đầy đủ cho
chuỗi muốn dự báo ("system" = chuỗi tổng, hoặc số zone 1..20).

Quy ước thời gian: cột h1 là khoảng 00:00–01:00, nên h1 -> 00:00 (nhãn đầu khoảng).
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

N_ZONES = 20
N_STATIONS = 11
HOUR_COLS = [f"h{i}" for i in range(1, 25)]

# Mốc dùng để CHỌN (zone đại diện, trạm nhiệt độ...). Chỉ được nhìn dữ liệu đến mốc này
# để không vô tình dùng thông tin của validation/test (khớp với src/evaluation.py).
SELECTION_END = pd.Timestamp("2006-06-30 23:00")


# ----------------------------------------------------------------------------
# 1. Phụ tải
# ----------------------------------------------------------------------------
def read_load_raw():
    """Đọc Load_history (số có dấu phẩy ngăn cách hàng nghìn) và Load_solution."""
    hist = pd.read_csv(RAW / "Load_history.csv", thousands=",")
    sol = pd.read_csv(RAW / "Load_solution.csv")
    sol = sol.drop(columns=["id", "weight"])
    return hist, sol


def wide_to_long(df, id_col):
    """Bảng rộng (1 dòng = 1 ngày, 24 cột giờ) -> bảng dài (1 dòng = 1 giờ)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    long = df.melt(id_vars=[id_col, "date"], value_vars=HOUR_COLS,
                   var_name="h", value_name="value")
    long["hour"] = long["h"].str[1:].astype(int) - 1          # h1 -> 0
    long["datetime"] = long["date"] + pd.to_timedelta(long["hour"], unit="h")
    return long[[id_col, "datetime", "value"]]


def missing_report(hist):
    """Mục 2.13: đếm số giờ thiếu của từng zone TRƯỚC khi ghép solution."""
    long = wide_to_long(hist, "zone_id")
    miss = long[long["value"].isna()]
    rep = (miss.groupby("zone_id")
               .agg(n_missing_hours=("value", "size"),
                    first_missing=("datetime", "min"),
                    last_missing=("datetime", "max")))
    rep = rep.reindex(range(1, N_ZONES + 1), fill_value=0)
    return rep


def build_zones_long():
    """Ghép history + solution, chuyển sang bảng dài, lấp các giờ còn thiếu bằng lag 168."""
    hist, sol = read_load_raw()
    h_long = wide_to_long(hist, "zone_id").rename(columns={"value": "load"})
    s_long = wide_to_long(sol[sol.zone_id <= N_ZONES], "zone_id").rename(columns={"value": "load_sol"})

    df = h_long.merge(s_long, on=["zone_id", "datetime"], how="left")
    df["filled_from_solution"] = df["load"].isna() & df["load_sol"].notna()
    df["load"] = df["load"].fillna(df["load_sol"])
    df = df.drop(columns="load_sol").sort_values(["zone_id", "datetime"]).reset_index(drop=True)

    # Các giờ vẫn còn thiếu (18 giờ ngày 30/06/2008, không có trong cả hai file)
    # -> lấp bằng giá trị cùng giờ tuần trước (lag 168), gắn cờ is_imputed.
    df["is_imputed"] = df["load"].isna()
    df["load"] = df.groupby("zone_id")["load"].transform(lambda s: s.fillna(s.shift(168)))
    assert df["load"].notna().all(), "vẫn còn giờ thiếu sau khi lấp lag 168"
    return df


def check_system_vs_zone21(zones_long):
    """Kiểm tra: tổng 20 zone phải khớp zone 21 trong Load_solution (9 tuần có đáp án)."""
    _, sol = read_load_raw()
    z21 = wide_to_long(sol[sol.zone_id == 21], "zone_id").set_index("datetime")["value"]
    total = zones_long.groupby("datetime")["load"].sum()
    diff = (total.reindex(z21.index) - z21).abs()
    return diff.max()


# ----------------------------------------------------------------------------
# 2. Nhiệt độ
# ----------------------------------------------------------------------------
def build_temperature():
    """11 trạm (°F). history kết thúc 30/06/2008 05:00, phần sau lấy từ temperature_solution."""
    th = pd.read_csv(RAW / "temperature_history.csv")
    t_long = wide_to_long(th, "station_id").dropna(subset=["value"])

    ts = pd.read_csv(RAW / "temperature_solution.csv")
    ts["datetime"] = (pd.to_datetime(ts["date"], format="%d%b%Y")
                      + pd.to_timedelta(ts["hour"] - 1, unit="h"))
    ts = ts.rename(columns={"T0_p1": "value"})[["station_id", "datetime", "value"]]

    t = pd.concat([t_long, ts]).drop_duplicates(["station_id", "datetime"], keep="first")
    wide = t.pivot(index="datetime", columns="station_id", values="value").sort_index()
    wide.columns = [f"T_s{c}" for c in wide.columns]
    wide["T_mean"] = wide[[f"T_s{i}" for i in range(1, N_STATIONS + 1)]].mean(axis=1)
    return wide


# ----------------------------------------------------------------------------
# 3. Lịch và ngày lễ
# ----------------------------------------------------------------------------
def read_holidays():
    """Holiday_List.csv: mỗi cột là một năm, ô dạng 'Monday, January 16'.
    Riêng một vài ô có ghi năm (vd 'Friday, December 31, 2004' = ngày nghỉ bù của Tết 2005)."""
    h = pd.read_csv(RAW / "Holiday_List.csv")
    h = h.rename(columns={h.columns[0]: "holiday"})
    rows = []
    for _, r in h.iterrows():
        for y in h.columns[1:]:
            cell = r[y]
            if pd.isna(cell):
                continue
            parts = [p.strip() for p in str(cell).split(",")]
            text = f"{parts[1]}, {parts[2]}" if len(parts) == 3 else f"{parts[1]}, {y}"
            d = pd.to_datetime(text, format="%B %d, %Y")
            rows.append({"holiday": r["holiday"], "date": d, "weekday_in_file": parts[0]})
    out = pd.DataFrame(rows)
    out["weekday_ok"] = out["date"].dt.day_name() == out["weekday_in_file"]
    return out


def build_calendar(index, holidays):
    cal = pd.DataFrame(index=index)
    cal["hour"] = index.hour
    cal["dayofweek"] = index.dayofweek          # 0 = thứ Hai
    cal["month"] = index.month
    cal["dayofyear"] = index.dayofyear
    cal["is_holiday"] = index.normalize().isin(holidays["date"]).astype(int)
    return cal


# ----------------------------------------------------------------------------
# 4. Chọn trạm nhiệt độ (2.11 cho chuỗi tổng, 2.15 cho từng zone)
# ----------------------------------------------------------------------------
def temp_r2(load, temp):
    """R² của hồi quy load ~ T + T² + T³ (chỉ dùng dữ liệu đến SELECTION_END)."""
    m = (load.index <= SELECTION_END)
    y = load[m].values
    t = temp.reindex(load.index)[m].values
    X = np.column_stack([np.ones_like(t), t, t ** 2, t ** 3])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - resid.var() / y.var()


def station_table(load, temperature):
    cols = [f"T_s{i}" for i in range(1, N_STATIONS + 1)] + ["T_mean"]
    return pd.Series({c: temp_r2(load, temperature[c]) for c in cols}).sort_values(ascending=False)


# ----------------------------------------------------------------------------
# 5. Chạy toàn bộ và lưu
# ----------------------------------------------------------------------------
def build_all(verbose=True):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    zones = build_zones_long()
    temp = build_temperature()
    hol = read_holidays()
    cal = build_calendar(temp.index, hol)

    zones.to_parquet(PROCESSED / "zones_long.parquet")
    temp.to_parquet(PROCESSED / "temperature.parquet")
    cal.to_parquet(PROCESSED / "calendar.parquet")

    # Gán trạm cho chuỗi tổng và từng zone: chọn cột có R² cao nhất (chỉ nhìn dữ liệu ≤ SELECTION_END)
    mapping = {}
    system = zones.groupby("datetime")["load"].sum()
    mapping["system"] = station_table(system, temp).index[0]
    for z in range(1, N_ZONES + 1):
        s = zones[zones.zone_id == z].set_index("datetime")["load"]
        mapping[str(z)] = station_table(s, temp).index[0]
    with open(PROCESSED / "zone_station.json", "w") as f:
        json.dump(mapping, f, indent=2)

    if verbose:
        print("zones_long :", zones.shape)
        print("temperature:", temp.shape, "| thiếu:", int(temp.isna().sum().sum()))
        print("holidays   :", len(hol), "ngày | thứ khớp:", hol["weekday_ok"].mean())
        print("chênh lệch tổng 20 zone vs zone 21:", check_system_vs_zone21(zones))
    return mapping


def load_dataset(series="system"):
    """
    Trả về bảng theo giờ (index = datetime) cho chuỗi cần dự báo:
        load        : phụ tải (kW)
        is_imputed  : giờ được lấp bằng lag 168 (không dùng để chấm điểm)
        T           : nhiệt độ dùng cho chuỗi này (trạm đã chọn ở zone_station.json)
        T_mean, T_s1..T_s11 : giữ lại để tham khảo
        hour, dayofweek, month, dayofyear, is_holiday : biến lịch (biết trước)
    """
    zones = pd.read_parquet(PROCESSED / "zones_long.parquet")
    temp = pd.read_parquet(PROCESSED / "temperature.parquet")
    cal = pd.read_parquet(PROCESSED / "calendar.parquet")
    with open(PROCESSED / "zone_station.json") as f:
        mapping = json.load(f)

    if series == "system":
        g = zones.groupby("datetime")
        df = pd.DataFrame({"load": g["load"].sum(), "is_imputed": g["is_imputed"].any()})
    else:
        z = zones[zones.zone_id == int(series)].set_index("datetime")
        df = z[["load", "is_imputed"]].copy()

    df = df.join(temp).join(cal)
    df["T"] = df[mapping[str(series)]]
    df.index.name = "datetime"
    return df


if __name__ == "__main__":
    print(build_all())
