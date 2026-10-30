# LSTM Gas Alarm / Time-to-Alarm

Báo cáo này mô tả dữ liệu `smoke_detection_iot.csv`, quy trình tiền xử lý,
mô hình LSTM và kết quả đánh giá phiên bản hiện tại.

> **Kết luận nhanh:** mô hình đã chạy thành công nhưng chất lượng dự đoán
> `time_to_alarm_sec` hiện **chưa đạt yêu cầu sử dụng thực tế**. Cần xử lý
> lại cách tạo nhãn và chia dữ liệu trước khi triển khai.

## 1. Cấu trúc thư mục

| Tệp | Mô tả |
|---|---|
| `data/raw/smoke_detection_iot.csv` | Dữ liệu cảm biến đầu vào |
| `src/eda_report.py` | Script thống kê và kiểm tra chất lượng dữ liệu |
| `reports/eda/*.csv` | Các báo cáo EDA |
| `src/train_lstm.py` | Script tạo nhãn, huấn luyện và lưu mô hình |
| `models/early_warning_lstm.pt` | LSTM phát hiện nguy cơ alarm trong 300 giây |
| `models/time_to_alarm_lstm.pt` | LSTM hồi quy số giây còn lại |
| `models/scaler_X.pkl` | Bộ chuẩn hóa các đặc trưng đầu vào |
| `models/scaler_y.pkl` | Bộ chuẩn hóa biến mục tiêu |
| `reports/evaluation_metrics.csv` | Metric test của hai giai đoạn |

## 2. Thông tin tổng quan dữ liệu

| Hạng mục | Kết quả |
|---|---:|
| Số dòng | 62.630 |
| Số cột | 16 |
| Kích thước tệp CSV | khoảng 5,56 MB |
| Dữ liệu thời gian | `UTC`, Unix timestamp |
| Khoảng thời gian UTC quan sát được | 417.864 giây |
| Khoảng cách kỳ vọng giữa các mẫu | 1 giây |
| Khoảng UTC khác 1 giây | 4 |
| Giá trị thiếu | 0 |
| Dòng trùng lặp | 0 |
| Số giá trị `Fire Alarm` | 2 (`0`, `1`) |

### Các cột dữ liệu

| Nhóm | Cột |
|---|---|
| Định danh/thời gian | `Unnamed: 0`, `UTC`, `CNT` |
| Đặc trưng dùng cho LSTM | `Temperature[C]`, `Humidity[%]`, `TVOC[ppb]`, `eCO2[ppm]`, `Pressure[hPa]` |
| Cảm biến khác chưa dùng | `Raw H2`, `Raw Ethanol`, `PM1.0`, `PM2.5`, `NC0.5`, `NC1.0`, `NC2.5` |
| Nhãn gốc | `Fire Alarm` |

## 3. Phân bố nhãn gốc

| `Fire Alarm` | Số dòng | Tỷ lệ |
|---:|---:|---:|
| 0 | 17.873 | 28,537% |
| 1 | 44.757 | 71,463% |
| **Tổng** | **62.630** | **100%** |

Nhãn bị lệch về lớp `1`. Tuy nhiên, đây là bài toán hồi quy
`time-to-alarm`, không phải chỉ là phân loại nhị phân, nên tỷ lệ nhãn gốc
không đủ để đánh giá chất lượng đồng hồ đếm ngược.

## 4. Thống kê mô tả các đặc trưng

| Đặc trưng | Trung bình | Độ lệch chuẩn | Nhỏ nhất | Trung vị | Lớn nhất |
|---|---:|---:|---:|---:|---:|
| `Temperature[C]` | 15,970 | 14,360 | -22,010 | 20,130 | 59,930 |
| `Humidity[%]` | 48,540 | 8,865 | 10,740 | 50,150 | 75,200 |
| `TVOC[ppb]` | 1.942 | 7.812 | 0 | 981 | 60.000 |
| `eCO2[ppm]` | 670 | 1.906 | 400 | 400 | 60.000 |
| `Raw H2` | 12.942 | 272 | 10.668 | 12.924 | 13.803 |
| `Raw Ethanol` | 19.754 | 610 | 15.317 | 19.501 | 21.410 |
| `Pressure[hPa]` | 938,628 | 1,331 | 930,852 | 938,816 | 939,861 |
| `PM1.0` | 100,594 | 922,524 | 0 | 1,810 | 14.334 |
| `PM2.5` | 184,468 | 1.976 | 0 | 1,880 | 45.432 |
| `NC0.5` | 491,464 | 4.266 | 0 | 12,450 | 61.482 |
| `NC1.0` | 203,587 | 2.215 | 0 | 1,943 | 51.915 |
| `NC2.5` | 80,049 | 1.083 | 0 | 0,044 | 30.026 |

Một số biến có giá trị cực đại lớn hơn nhiều so với trung vị, đặc biệt là
`TVOC`, `eCO2`, `PM1.0`, `PM2.5` và các cột `NC`. Cần kiểm tra đây là giá trị
cảm biến hợp lệ hay lỗi đo trước khi dùng các biến này trong mô hình.

## 5. Tạo nhãn đếm ngược (đã sửa)

Script duyệt dữ liệu từ cuối về đầu trong từng đoạn UTC liên tục:

```text
time_to_alarm[i] = UTC của Fire Alarm = 1 tiếp theo - UTC hiện tại
```

Sau đó tạo cửa sổ:

| Hạng mục | Giá trị |
|---|---:|
| Độ dài cửa sổ | 30 mẫu |
| Ý nghĩa dự kiến | 30 giây lịch sử |
| Số cửa sổ hợp lệ | 51.052 |
| Min nhãn | 0 |
| Max nhãn | 3.178 |
| Trung bình nhãn | 197,52 |
| Trung vị nhãn | 0 |

Các dòng sau sự kiện cảnh báo cuối cùng trong một đoạn được bỏ qua vì không
có sự kiện tương lai để định nghĩa thời gian đếm ngược. Các cửa sổ đi qua
khoảng UTC bị gián đoạn cũng bị loại.

## 6. Phiên bản hai giai đoạn đã triển khai

### Giai đoạn 1: phát hiện cảnh báo sớm

Nhãn dương là các cửa sổ có `time_to_alarm_sec` trong khoảng `1–300` giây.
Mô hình dùng `BCEWithLogitsLoss` với `pos_weight` tự động để xử lý mất cân
bằng và được đánh giá bằng Precision, Recall, F1 và PR-AUC.

### Giai đoạn 2: hồi quy thời gian

Chỉ các mẫu có thời gian đến alarm lớn hơn 0 được đưa vào hồi quy. Mô hình
dùng Weighted Huber Loss, trong đó các mẫu gần alarm được ưu tiên hơn và ít
nhạy với outlier hơn MSE.

| Thành phần | Cấu hình |
|---|---|
| Đặc trưng đầu vào | 5 cột cảm biến |
| Chuẩn hóa | `MinMaxScaler` |
| Kiến trúc | LSTM 2 lớp |
| Hidden size | 64 |
| Dropout | 0,2 |
| Fully-connected | 64 → 32 → 1 |
| Hàm mất mát | Weighted BCE + Weighted Huber |
| Tối ưu hóa | Adam |
| Learning rate | 0,001 |
| Batch size | 64 |
| Epochs | 10 |
| Validation | 20% của phần train |
| Early stopping | Patience = 3 |
| Cách chia | Theo sự kiện alarm, không chia ngẫu nhiên từng dòng |
| Thiết bị chạy | CPU |

## 7. Chia dữ liệu và đánh giá

Các cửa sổ được chia theo các sự kiện alarm độc lập. Scaler chỉ được fit trên
phần train:

| Tập | Số cửa sổ | Phạm vi nhãn |
|---|---:|---:|
| Test classifier | 1.124 | positive rate 0,267% |
| Test regression | 3 | 0 đến 3.148 giây |

### Kết quả trên test

| Chỉ số | Kết quả |
|---|---:|
| Precision | 0,0044 |
| Recall | 1,0000 |
| F1 | 0,0087 |
| PR-AUC | 0,0032 |
| Regression MAE | 1.414,50 giây |
| Regression RMSE | 1.414,53 giây |
| Dự đoán hồi quy âm | 0% |

Kết quả này chưa đủ để triển khai thực tế. Recall cao nhưng Precision rất thấp,
nghĩa là mô hình phát hiện được mẫu dương hiếm hoi nhưng báo động giả nhiều.
Metric hồi quy không đáng tin cậy vì test chỉ có 3 mẫu dương. Đây là giới hạn
của dữ liệu, không phải lý do để dùng random oversampling hoặc SMOTE làm đẹp
đánh giá.

## 8. Những điểm đã xử lý

1. Nhãn dùng chênh lệch `UTC` thực tế.
2. Tách logic theo các đoạn UTC liên tục và bỏ cửa sổ qua gap.
3. Scaler chỉ fit trên phần train.
4. Có validation, early stopping và `pos_weight`.
5. Dùng Weighted Huber cho hồi quy và đánh giá PR-AUC.

## 9. Hạn chế còn lại

1. Tập test theo sự kiện vẫn mất cân bằng nghiêm trọng.
2. Một số cảm biến có outlier rất lớn cần xác minh với tài liệu phần cứng.
3. Chưa có đủ các sự kiện cháy độc lập để đánh giá tổng quát hóa.
4. Số sự kiện độc lập quá ít để kết luận khả năng tổng quát hóa.
5. Chưa nên triển khai cảnh báo tự động khi chưa có thêm dữ liệu.

## 10. Khuyến nghị phiên bản tiếp theo

| Ưu tiên | Việc cần làm |
|---:|---|
| 1 | Thu thập thêm nhiều sự kiện alarm độc lập |
| 2 | Bổ sung mẫu tiền cảnh báo trước khi `Fire Alarm=1` |
| 3 | Chọn threshold theo mục tiêu giảm false alarm |
| 4 | Báo cáo MAE theo các dải 0–10, 10–60 và trên 60 giây |
| 5 | Chỉ triển khai sau khi test có đủ sự kiện dương |

## 10. Cách chạy lại

```powershell
python src\eda_report.py
python src\train_lstm.py
```

Các báo cáo EDA được sinh lại gồm:

- `reports/eda/eda_numeric_summary.csv`
- `reports/eda/eda_alarm_distribution.csv`
- `reports/eda/eda_utc_gaps.csv`
