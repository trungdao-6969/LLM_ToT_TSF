# Hướng dẫn chạy và đọc hiểu (GĐ2 → GĐ5)

Gói này làm xong mọi thứ chạy được mà **chưa cần LLM**:
- sửa GĐ2 và GĐ3 theo nhận xét của thầy;
- toàn bộ baseline GĐ4 ở 3 mức nhiệt độ;
- bộ chọn thống kê GĐ5, tức đối thủ mà ToT phải thắng.

Riêng GĐ6 (ToT) cần em chọn API ở mục 0.3 trước.

Anh đã chạy toàn bộ trên dữ liệu thật trước khi gửi. Các con số ở mục 4 là kết quả anh nhận được, em dùng để đối chiếu khi tự chạy lại. LightGBM có cố định seed, nên kết quả của em phải gần như trùng khớp.

---

## 1. Chuẩn bị (10 phút)

1. **Tạo nhánh git mới** để không đè lên code cũ:
   ```bash
   cd D:\Project_ThayDu\LLM_ToT_TSF
   git checkout -b v2-theo-nhan-xet
   ```
2. **Giải nén** gói này vào thư mục repo. Các file trùng tên trong `src/` và `README.md` sẽ được thay mới. Các notebook cũ (01_load_data, 04a, 04b…) vẫn còn, nhưng **có thể không chạy được nữa** vì `src/` đã đổi. Nếu muốn giữ để tham khảo, hãy chuyển chúng vào `notebooks/old/`.
3. **Copy dữ liệu:** 6 file trong `GEFCOM2012_Data/Load/` → `data/raw/`.
4. **Cài thư viện:**
   ```bash
   conda activate tot_tsf
   pip install -r requirements.txt
   ```

## 2. Thứ tự chạy

| Notebook | Checklist | Thời gian lần đầu | Kiểm tra nhanh |
|---|---|---|---|
| 01_du_lieu | 2.1–2.13, 2.16 | < 1 phút | chênh lệch tổng 20 zone với zone 21 = 0 |
| 02_eda_chon_zone | 2.14–2.15 | < 1 phút | zone chọn được: 4 / 9 / 10 |
| 03_giao_thuc_danh_gia | 3.1–3.8 | 2–5 phút | "Tất cả đạt: True" ở bảng kiểm tra rò rỉ |
| 04_du_bao_nhiet_do | 4.0 | < 1 phút | regression có MAE ≈ 3,8 °F |
| 05_baselines | 4.1–4.6 | **khoảng 60 phút** | xem mục 4 |
| 06_bo_chon_thong_ke | 5.1–5.8 | **khoảng 60 phút** (24 cấu hình × 4 chuỗi) | xem mục 4 |

Notebook 05 và 06 lưu dự báo vào `results/forecasts/`. Lần chạy thứ hai sẽ đọc lại file nên chỉ mất vài giây. Muốn chạy lại từ đầu thì xóa thư mục đó.

Lời khuyên: **chạy 01 → 04 trước, đọc kỹ từng cell**. Sau đó bật 05 và 06 chạy qua đêm.

## 3. Đọc code theo thứ tự nào

Mỗi file anh viết theo đúng cách mình vẫn học: từ bài toán suy ra cần file gì.

1. **`src/evaluation.py`, hàm `information_set`.** Đây là trái tim của cả project. Mọi mô hình chỉ được nhìn dữ liệu qua hàm này. Tương lai bị **xóa thật** (gán NaN), nên mô hình nào lỡ dùng thông tin tương lai sẽ ra NaN và lỗi ngay, chứ không lặng lẽ cho kết quả đẹp giả.
2. **`src/evaluation.py`, hàm `backtest`.** Vòng lặp qua các origin, fit lại đầu mỗi tháng.
3. **`src/models.py`.** Đọc `SeasonalNaive` trước (5 dòng), rồi `MLRTaoHong`: hàm `design` chính là phương trình Tao Hong em đã học, viết bằng ma trận. Để ý hàm `temp_column`: đây là chỗ duy nhất phân biệt L1 với L2/L3.
4. **`src/selectors.py`.** Tất cả bộ chọn đều xoay quanh hàm `score`, cũng là hàm ToT sẽ dùng để chấm thought.

## 4. Kết quả anh nhận được (để em đối chiếu)

**Baseline chuỗi tổng, MAPE test (%)** (notebook 05):

| Mô hình | L1 (bản chính) | L2 | L3 (oracle) |
|---|---|---|---|
| Seasonal naive 24 | 8,04 | – | – |
| MSTL + ETS | 8,56 | – | – |
| MLR Tao Hong | 8,05 | 6,36 | 5,11 |
| SARIMAX | 6,73 | 6,03 | 5,10 |
| LightGBM | **6,48** | **5,32** | **3,58** |

Dự báo nhiệt độ cho L2: MAE ≈ 3,8 °F (notebook 04).

**Bộ chọn thống kê, chuỗi tổng, L1, MAPE test (%)** (notebook 06):

| Bộ chọn | MAPE |
|---|---|
| Chọn tĩnh (tốt nhất trên validation) | 6,39 |
| Chọn động S1 | 6,46 |
| Random search 9 cấu hình (TB 20 seed) | 6,46 ± 0,08 |
| Oracle tháng (cận trên khi chọn theo tháng) | 6,18 |
| Oracle ngày (cận trên khi chọn theo ngày) | 3,86 |

**Ba zone đại diện, WAPE test (%)** (notebook 06, mục 5.8):

| | Chọn tĩnh | Chọn động S1 | Random 9 | Oracle tháng |
|---|---|---|---|---|
| Zone 4 (ổn định) | 6,58 | 6,58 | 6,71 | 6,54 |
| Zone 9 (biến động) | 17,69 | 20,08 | 18,95 | 17,10 |
| Zone 10 (đổi chế độ) | 8,56 | **14,80** | 13,41 | 8,08 |

**Ba phát hiện quan trọng nhất để báo thầy:**
1. **Bỏ giả định nhiệt độ thật làm sai số tăng gần gấp đôi.** LightGBM: 3,58% ở L3, 6,48% ở L1. Trước đây kết quả của mình đẹp hơn thực tế.
2. **Trên chuỗi tổng, chọn tĩnh đã rất sát trần chọn-theo-tháng** (6,39% so với 6,18%). ToT chỉ còn khoảng 0,2 điểm % để thắng nếu vẫn quyết định theo tháng.
3. **Ở zone 10, chọn động dựa trên quá khứ thất bại sau khi đổi chế độ** (14,8% so với 8,6%). Lý do: tập chấm điểm toàn là ngày trước bước nhảy. Đây là chỗ giả thuyết của LLM có thể tạo khác biệt thật.

⚠ **Không chạy hai notebook nặng (05, 06) cùng lúc.** LightGBM dùng nhiều luồng; hai tiến trình tranh nhau CPU sẽ chậm đi hàng chục lần (anh đã gặp khi chạy thử).

Ở zone, đọc **WAPE** thay vì MAPE: zone 4 và 9 có giờ phụ tải ≈ 0 làm MAPE bị nổ (lý do ghi ở notebook 05).

## 5. Những quyết định anh đã chọn thay em (em cần hiểu để bảo vệ trước thầy)

| Quyết định | Lý do | Cần thầy xác nhận? |
|---|---|---|
| L1 dùng nhiệt độ **trễ 24 giờ** và thống kê nhiệt độ ngày D−1 làm đặc trưng | Đây là thông tin đã quan sát tại origin. Không có mô hình dự báo nhiệt độ riêng, nên khác L2 | ✔ (câu 2) |
| L2 dùng mô hình hồi quy nhiệt độ tự xây; fit vẫn dùng nhiệt độ thật quá khứ | Không có dữ liệu dự báo thời tiết lịch sử; đây là cách ex-ante thông dụng | ✔ (câu 3) |
| Zone 10 được chọn nhờ nhìn cả giai đoạn | Chọn chuỗi là thiết kế thí nghiệm, không phải đầu vào mô hình | nên báo thầy |
| SARIMAX chỉ làm baseline, không có trong thực đơn của LLM | Ước lượng chậm; mỗi lần quyết định ToT phải chấm 9 cấu hình × hàng chục ngày | nên báo thầy |
| Origin từ 07/2005 đến 06/2006 chỉ dùng để chấm điểm | Đầu validation, S1 cần 365 ngày quá khứ đã có dự báo | không |
| 41 origin đại diện cho mỗi giai đoạn (tối đa 2 ngày mỗi tầng; một số tầng ngày lễ có ít ngày) | Con số tạm, chờ thầy | ✔ (câu 4) |
| Zone đọc theo WAPE, chuỗi tổng giữ MAPE | Zone 4, 9 có giờ phụ tải ≈ 0 | nên báo thầy |
| Cách chấm mặc định cho ToT: S1 | Độ tin cậy cao nhất trên validation (Spearman 0,87, regret 0,18) | không |

## 6. Câu hỏi tự kiểm tra (nên trả lời được trước khi báo cáo thầy)

1. Vì sao ở mức L1, MLR Tao Hong tụt hạng mạnh nhất so với L3, còn Seasonal Naive và MSTL không thay đổi gì?
2. Trong `LightGBMModel`, tùy chọn `target="ratio"` giúp gì cho zone 10? Gợi ý: cây quyết định không ngoại suy được ra ngoài khoảng giá trị đã thấy khi train.
3. Kết quả chọn động có tốt hơn chọn tĩnh không? Nếu không, đó là tin tốt hay tin xấu cho ToT?
4. Khoảng cách giữa bộ chọn tốt nhất và **oracle tháng** cho biết điều gì về "trần" mà ToT có thể đạt được?
5. Vì sao random search phải lặp 20 seed, và ToT có cần làm điều tương tự không?

## 7. Bước tiếp theo

1. Gửi thầy: bảng kết quả GĐ4 (`results/tables/4_6_*.csv`, hình `4_6_baselines_test.png`), bảng GĐ5 (`5_*.csv`), 3 phát hiện ở mục 4 và các câu hỏi trong checklist. Đây là **báo cáo thầy lần 1**.
2. Hỏi thêm thầy một câu mới: **ToT nên quyết định theo tháng hay theo từng origin đại diện?** Kết quả GĐ5 cho thấy nếu quyết định theo tháng thì gần như không còn chỗ để thắng trên chuỗi tổng.
3. Chọn API LLM (mục 0.3) để bắt đầu GĐ6.
