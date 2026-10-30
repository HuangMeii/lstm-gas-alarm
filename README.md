# NIST Smoke Alarm Performance Pipeline

Pipeline LSTM hai giai đoạn sử dụng dữ liệu chính thức
[NIST – Data on Smoke Alarm Performance](https://www.nist.gov/el/fire/engineered-fire-safety/data-smoke-alarm-performance).

Dataset Kaggle cũ được giữ trong `data/raw/legacy/` để tái lập, nhưng không
còn là đầu vào mặc định.

## Cấu trúc dự án

```text
data/
├── raw/
│   ├── nist/
│   │   ├── archives/       # ZIP/XLSX tải từ NIST
│   │   ├── room/           # Room experiments
│   │   └── smokebox/       # Smoke-box experiments
│   └── legacy/             # Dataset Kaggle cũ
└── processed/nist/         # Manifest đã chuẩn hóa
models/
├── nist/                   # Model/scaler NIST
└── legacy/                 # Artifact Kaggle cũ
reports/
├── nist/                   # Metric NIST
└── legacy/                 # Báo cáo Kaggle cũ
src/
├── nist_pipeline/          # Config, ingestion, dataset windows
└── legacy/                 # Code Kaggle cũ
scripts/
├── prepare_nist.py         # Tạo manifest NIST
└── train_nist.py           # Huấn luyện pipeline NIST
```

## Dữ liệu NIST

Pipeline sử dụng:

- `roomdata.zip`
- `smokeboxdata.zip`
- `summary_nistsmokebox_roomresults.xlsx`

Mỗi experiment được xử lý độc lập. Không nối các file thành một chuỗi duy
nhất. Các đặc trưng chung được dùng là:

- `obscuration`
- `mic`

Nhãn được tính theo:

```text
time_to_alarm_sec = alarm_time_sec - time_sec
```

Dữ liệu sau alarm được loại bỏ. Alarm của smoke-box lấy từ cột `Alarm?`; alarm
của room lấy từ workbook tổng hợp NIST.

## Mô hình

### Giai đoạn 1 — Early warning

Dự đoán alarm có xảy ra trong 300 giây tới hay không.

- LSTM
- `BCEWithLogitsLoss`
- `pos_weight` tự động
- Precision, Recall, F1, PR-AUC

### Giai đoạn 2 — Time-to-alarm

Dự đoán số giây còn lại tới alarm trên các mẫu trước alarm.

- LSTM
- Huber Loss
- MAE theo giây

Train/validation/test được chia theo `experiment_id`, không chia ngẫu nhiên
từng dòng, nhằm tránh leakage giữa các cửa sổ liên tiếp.

## Kết quả sơ bộ NIST

Kết quả được lưu tại `reports/nist/evaluation_metrics.csv`.

| Metric | Kết quả |
|---|---:|
| Tổng experiment có window | 477 |
| Experiment test | 72 |
| Classification test windows | 19.388 |
| Precision | 0,867 |
| Recall | 0,310 |
| F1 | 0,457 |
| PR-AUC | 0,857 |
| Regression test windows | 19.316 |
| Regression MAE | 150,33 giây |

Đây là kết quả sơ bộ, chưa phải chứng nhận hiệu năng triển khai. Cần đánh giá
thêm theo từng loại fire scenario, alarm model, unit và room/smoke-box setup.

## Cài đặt

```powershell
python -m pip install -r requirements.txt
```

## Chạy pipeline

Từ thư mục gốc repository:

```powershell
python scripts\prepare_nist.py
python scripts\train_nist.py
```

## Artifact đầu ra

- `data/processed/nist/experiment_manifest.csv`
- `models/nist/early_warning_lstm.pt`
- `models/nist/time_to_alarm_lstm.pt`
- `models/nist/nist_feature_scaler.pkl`
- `models/nist/nist_target_scaler.pkl`
- `reports/nist/evaluation_metrics.csv`

## Giới hạn

1. NIST là dữ liệu thí nghiệm có kiểm soát, không đại diện cho mọi môi trường
   dân dụng.
2. Các room experiment có nhiều alarm model/unit; manifest giữ metadata theo
   experiment nhưng phiên bản baseline hiện chọn một alarm time đại diện cho
   mỗi file room.
3. Chưa đủ để kết luận mô hình dùng cho cảnh báo an toàn thực tế.
4. Cần thêm calibration threshold, đánh giá false-alarm rate và kiểm thử theo
   experiment hoàn toàn chưa xuất hiện trong train.
