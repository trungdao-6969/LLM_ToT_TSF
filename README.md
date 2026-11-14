# LLM_ToT_TSF – Tree of Thoughts cho dự báo phụ tải điện

Đồ án: dùng LLM theo khung **Tree of Thoughts** để đề xuất giả thuyết về chuỗi thời gian và cấu hình mô hình dự báo, còn việc **chấm điểm** do dữ liệu đảm nhận (sai số backtest trên các giai đoạn lịch sử tương tự), thay vì để LLM tự đánh giá. Phương pháp được so sánh với các cách chọn mô hình thông thường.

- **Dữ liệu:** GEFCom2012, track Load: 20 zone, 11 trạm nhiệt độ, theo giờ, 2004–2008.
- **Bài toán:** dự báo ngày tới. Origin là 00:00 ngày D; dự báo 24 giờ của ngày D; phụ tải biết đến 23:00 ngày D−1.

## Cấu trúc thư mục

```
data/raw/            6 file CSV gốc của track Load (không đưa lên git)
data/processed/      dữ liệu đã xử lý (sinh ra bởi notebook 01)
src/
  data.py            xử lý dữ liệu, load_dataset(series)
  eda.py             đặc trưng zone, chọn zone đại diện
  evaluation.py      giao thức đánh giá dùng chung (information set 3 mức nhiệt độ, backtest,
                     metric, kiểm tra rò rỉ, S1/S2/S3, DM test, origin đại diện)
  temperature.py     dự báo nhiệt độ cho mức L2
  models.py          baseline + thực đơn cấu hình (SEARCH_SPACE)
  experiments.py     chạy backtest có cache
  selectors.py       bộ chọn mô hình thống kê (GĐ5)
notebooks/
  01_du_lieu.ipynb               GĐ2
  02_eda_chon_zone.ipynb         GĐ2 (zone đại diện)
  03_giao_thuc_danh_gia.ipynb    GĐ3
  04_du_bao_nhiet_do.ipynb       GĐ4 – mục 4.0
  05_baselines.ipynb             GĐ4
  06_bo_chon_thong_ke.ipynb      GĐ5
results/tables, results/figures  bảng và hình cho báo cáo
```

## Cài đặt

```bash
conda activate tot_tsf
pip install -r requirements.txt
```

Copy 6 file trong `GEFCOM2012_Data/Load/` vào `data/raw/`, rồi chạy các notebook theo thứ tự 01 → 06.

## Ba mức nhiệt độ

| Mức | Nhiệt độ ngày D mà mô hình được thấy | Vai trò |
|---|---|---|
| L1 | không thấy (chỉ đến 23:00 D−1) | bản chính |
| L2 | nhiệt độ dự báo (`src/temperature.py`) | gần vận hành thực tế |
| L3 | nhiệt độ thật | oracle experiment |
