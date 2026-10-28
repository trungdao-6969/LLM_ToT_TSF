"""
Xử lý dữ liệu GEFCom2012 (track Load) cho đồ án ToT/GoT dự báo phụ tải điện.

Module này gom toàn bộ các bước đã làm từng bước trong ba notebook của Giai đoạn 2
thành các hàm có thể gọi lại:

    notebooks/01_load_data.ipynb            -> load_zones_long(), system_from_zones()
    notebooks/02_temperature_holidays.ipynb -> load_temperature(), load_holidays(),
                                               add_calendar_features(), build_master()
    notebooks/03_eda.ipynb                  -> (phân tích, không có hàm riêng)

Các notebook ở giai đoạn sau (đánh giá, baseline, ToT) chỉ cần:

    from src.data import load_dataset
    df = load_dataset()

Quy ước thời gian dùng trong toàn bộ đồ án
-------------------------------------------
Mỗi cột hk (k = 1..24) trong file gốc là phụ tải / nhiệt độ của KHOẢNG một giờ
từ (k-1):00 đến k:00. Ta gắn nhãn bằng GIỜ BẮT ĐẦU khoảng ("nhãn đầu khoảng"):
    h1  -> 00:00,  h2 -> 01:00,  ...,  h24 -> 23:00 cùng ngày.
Bằng chứng: file temperature_solution ghi giờ thứ 24 của ngày 07/07/2008 là
"08Jul2008:0:00:00", tức hk là giờ KẾT THÚC lúc k:00.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
HOUR_COLS = [f"h{i}" for i in range(1, 25)]   # 24 cột giờ trong file gốc
DATE_COLS = ["year", "month", "day"]
LOAD_KEYS = ["zone_id"] + DATE_COLS           # khóa định danh một dòng phụ tải
N_ZONES = 20                                  # số zone trong Load_history
N_STATIONS = 11                               # số trạm nhiệt độ
SYSTEM_ZONE_ID = 21                           # zone 21 trong Load_solution = tổng hệ thống

# Thư mục mặc định, tính từ vị trí file này: <repo>/src/data.py -> <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------
def wide_to_long(df: pd.DataFrame, id_cols: list, value_name: str) -> pd.DataFrame:
    """
    Chuyển bảng rộng (một dòng = một ngày, 24 cột h1..h24) sang bảng dài
    (một dòng = một giờ), kèm cột `datetime` theo quy ước nhãn đầu khoảng.

    Tham số
    -------
    df         : bảng rộng, phải có các cột id_cols, year, month, day, h1..h24
    id_cols    : các cột định danh giữ nguyên (ví dụ ["zone_id", "year", "month", "day"])
    value_name : tên cột giá trị trong bảng dài (ví dụ "load" hoặc "temp")

    Trả về
    ------
    Bảng dài gồm các cột id_cols (trừ year/month/day) + "datetime" + value_name.
    """
    long = df.melt(id_vars=id_cols, value_vars=HOUR_COLS,
                   var_name="hour", value_name=value_name)
    hour = long["hour"].str[1:].astype(int)                  # "h7" -> 7
    long["datetime"] = (pd.to_datetime(long[DATE_COLS])      # 00:00 của ngày
                        + pd.to_timedelta(hour - 1, unit="h"))  # + (k-1) giờ
    keep = [c for c in id_cols if c not in DATE_COLS] + ["datetime", value_name]
    return long[keep]


# ---------------------------------------------------------------------------
# Phụ tải (notebook 01)
# ---------------------------------------------------------------------------
def load_zones_long(raw_dir=DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Đọc phụ tải 20 zone, lấp 9 tuần bị xóa bằng Load_solution, chuyển sang bảng
    dài và lấp 18 giờ mất hẳn ngày 30/06/2008 bằng giá trị cùng giờ tuần trước.

    Trả về bảng dài các cột: zone_id, datetime, load, is_imputed
    (792 000 dòng = 20 zone x 1 650 ngày x 24 giờ).
    """
    raw_dir = Path(raw_dir)

    # 1) Đọc file gốc. Số phụ tải có dấu phẩy hàng nghìn ("16,853") nên cần thousands=","
    history = pd.read_csv(raw_dir / "Load_history.csv", thousands=",")
    solution = pd.read_csv(raw_dir / "Load_solution.csv", thousands=",")

    # 2) Giữ zone 1..20 của file solution, bỏ cột id (số thứ tự) và weight (trọng số chấm điểm)
    solution = solution[solution["zone_id"] <= N_ZONES].drop(columns=["id", "weight"])

    # 3) Điền ô trống của history bằng ô cùng (zone, ngày, giờ) trong solution
    full = (history.set_index(LOAD_KEYS)
                   .fillna(solution.set_index(LOAD_KEYS))
                   .reset_index())

    # 4) Bảng rộng -> bảng dài, sắp theo zone rồi theo thời gian
    zones = (wide_to_long(full, LOAD_KEYS, "load")
             .sort_values(["zone_id", "datetime"])
             .reset_index(drop=True))

    # 5) Đánh dấu rồi lấp các giờ còn thiếu bằng giá trị cùng giờ tuần trước (lag 168).
    #    Đánh dấu TRƯỚC khi lấp, để sau này loại các giờ này khỏi mọi phép chấm điểm.
    zones["is_imputed"] = zones["load"].isna()
    zones["load"] = zones["load"].fillna(zones.groupby("zone_id")["load"].shift(168))

    if zones["load"].isna().any():
        raise ValueError("Vẫn còn giá trị phụ tải bị thiếu sau khi lấp.")
    return zones


def system_from_zones(zones: pd.DataFrame) -> pd.DataFrame:
    """
    Cộng 20 zone theo từng giờ để được chuỗi tổng hệ thống
    (đúng định nghĩa zone 21 của cuộc thi).

    Trả về các cột: datetime, load, is_imputed.
    """
    system = (zones.groupby("datetime")
                   .agg(load=("load", "sum"),
                        n_zones=("load", "count"),
                        is_imputed=("is_imputed", "any"))
                   .reset_index())
    # pandas bỏ qua NaN khi cộng, nên phải kiểm tra mỗi giờ có đủ 20 zone
    if not (system["n_zones"] == N_ZONES).all():
        raise ValueError("Có giờ không đủ 20 zone, chuỗi tổng sẽ bị thấp hơn thực tế.")
    return system.drop(columns="n_zones")


# ---------------------------------------------------------------------------
# Nhiệt độ (notebook 02)
# ---------------------------------------------------------------------------
def load_temperature(raw_dir=DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Đọc nhiệt độ 11 trạm (đơn vị °F), ghép phần lịch sử với phần solution
    (từ 06:00 ngày 30/06/2008 đến 23:00 ngày 07/07/2008).

    Trả về bảng rộng theo trạm: datetime, T1, ..., T11, T_mean
    (T_mean = trung bình 11 trạm).
    """
    raw_dir = Path(raw_dir)

    # Phần lịch sử: cùng định dạng bảng rộng như phụ tải (station_id thay cho zone_id)
    hist = pd.read_csv(raw_dir / "temperature_history.csv")
    hist_long = wide_to_long(hist, ["station_id"] + DATE_COLS, "temp").dropna(subset=["temp"])

    # Phần solution: đã ở dạng dài, cột `hour` = k của hk (1..24), giá trị ở cột T0_p1
    sol = pd.read_csv(raw_dir / "temperature_solution.csv")
    sol_long = pd.DataFrame({
        "station_id": sol["station_id"],
        "datetime": pd.to_datetime(sol[DATE_COLS]) + pd.to_timedelta(sol["hour"] - 1, unit="h"),
        "temp": sol["T0_p1"].astype(float),
    })

    temp_long = pd.concat([hist_long, sol_long], ignore_index=True)
    if temp_long.duplicated(["station_id", "datetime"]).any():
        raise ValueError("Phần lịch sử và phần solution của nhiệt độ bị trùng giờ.")

    # Bảng dài -> bảng rộng: mỗi trạm một cột T1..T11
    temp = temp_long.pivot(index="datetime", columns="station_id", values="temp")
    temp.columns = [f"T{s}" for s in temp.columns]
    temp["T_mean"] = temp.mean(axis=1)
    return temp.reset_index()


# ---------------------------------------------------------------------------
# Ngày lễ (notebook 02)
# ---------------------------------------------------------------------------
def load_holidays(raw_dir=DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Đọc Holiday_List.csv (ngày ghi dạng chữ, ví dụ "Monday, July 5") thành bảng
    các cột: date, holiday.

    Lưu ý: đây là ngày NGHỈ THỰC TẾ (đã tính nghỉ bù). Ví dụ New Year's Day 2005
    được nghỉ vào thứ Sáu 31/12/2004, nên ô của cột 2005 ghi "Friday, December 31, 2004".
    """
    raw = pd.read_csv(Path(raw_dir) / "Holiday_List.csv", index_col=0)
    rows = []
    for holiday, by_year in raw.iterrows():
        for year, text in by_year.items():
            if pd.isna(text):               # năm 2008 chỉ có dữ liệu đến tháng 7
                continue
            parts = [p.strip() for p in str(text).split(",")]
            # parts = ["Monday", "July 5"]  hoặc  ["Friday", "December 31", "2004"]
            month_day = parts[1]
            real_year = parts[2] if len(parts) == 3 else year
            date = pd.to_datetime(f"{month_day} {real_year}", format="%B %d %Y")
            rows.append({"date": date, "holiday": holiday})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Thêm các đặc trưng lịch từ cột `datetime`."""
    out = df.copy()
    dt = out["datetime"].dt
    out["hour"] = dt.hour                       # 0..23 (giờ bắt đầu khoảng)
    out["dayofweek"] = dt.dayofweek             # 0 = thứ Hai, ..., 6 = Chủ nhật
    out["month"] = dt.month
    out["year"] = dt.year
    out["dayofyear"] = dt.dayofyear
    out["is_weekend"] = out["dayofweek"] >= 5
    out["is_holiday"] = dt.normalize().isin(holidays["date"])
    return out


# ---------------------------------------------------------------------------
# Bảng tổng (notebook 02) và điểm vào chính
# ---------------------------------------------------------------------------
def build_master(raw_dir=DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Ghép chuỗi tổng hệ thống + nhiệt độ + đặc trưng lịch thành một bảng duy nhất,
    mỗi dòng là một giờ (39 600 dòng).
    """
    system = system_from_zones(load_zones_long(raw_dir))
    temp = load_temperature(raw_dir)
    master = system.merge(temp, on="datetime", how="left", validate="one_to_one")
    if master["T_mean"].isna().any():
        raise ValueError("Có giờ phụ tải không có nhiệt độ tương ứng.")
    return add_calendar_features(master, load_holidays(raw_dir))


def load_dataset(processed_dir=DEFAULT_PROCESSED_DIR, raw_dir=DEFAULT_RAW_DIR,
                 rebuild: bool = False) -> pd.DataFrame:
    """
    Trả về bảng tổng của đồ án. Lần đầu (hoặc khi rebuild=True) sẽ xây từ dữ liệu
    thô rồi lưu ra data/processed/master_system.parquet; các lần sau chỉ đọc file đó.
    """
    path = Path(processed_dir) / "master_system.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    master = build_master(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(path, index=False)
    return master
