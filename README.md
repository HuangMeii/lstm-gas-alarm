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

## Calibration và đánh giá vận hành

Pipeline dùng thứ tự:

```text
group-based split
→ probability calibration (Platt scaling)
→ threshold sweep trên validation
→ debounce 3/5 cửa sổ
→ event-level false-alarm evaluation
→ unseen-scenario test
→ regression evaluation theo horizon
```

Threshold không được chọn trên test. Quy tắc hiện tại là chọn threshold có
precision cao nhất trong validation với recall tối thiểu 0,90.

Artifact đánh giá:

- `reports/nist/calibration_metrics.csv`
- `reports/nist/threshold_sweep.csv`
- `reports/nist/selected_threshold.json`
- `reports/nist/false_alarm_metrics.csv`
- `reports/nist/experiment_predictions.csv`
- `reports/nist/regression_metrics_by_horizon.csv`
- `models/nist/nist_probability_calibrator.pkl`

## Kết quả NIST sau calibration

Kết quả được lưu tại `reports/nist/evaluation_metrics.csv`.

| Metric | Kết quả |
|---|---:|
| Tổng experiment có window | 487 |
| Scenario test chưa xuất hiện trong train | 41 |
| Experiment test | 78 |
| Classification test windows | 20.262 |
| Calibration Brier score | 0,2164 |
| Calibration ECE | 0,0463 |
| Threshold | 0,62 |
| Precision | 0,753 |
| Recall | 1,000 |
| F1 | 0,859 |
| PR-AUC | 0,773 |
| Regression test windows | 19.018 |
| Regression MAE | 151,67 giây |
| Event recall | 1,000 |
| False-alarm rate / experiment | 100% (1/1) |
| False alarms / hour | 3,09 |

Kết quả **chưa đạt triển khai**: recall cao nhưng false-alarm rate là 100% trên
experiment âm duy nhất trong test. Mẫu âm quá ít nên con số này chưa ổn định,
nhưng vẫn là cảnh báo bắt buộc: cần thêm experiment không alarm và không được
đánh giá model chỉ bằng F1/PR-AUC.

### Regression theo horizon

| Khoảng thật | Số window | MAE |
|---|---:|---:|
| 0–10 giây | 770 | 227,09 giây |
| 10–30 giây | 1.540 | 215,83 giây |
| 30–60 giây | 2.283 | 197,33 giây |
| 60–300 giây | 10.478 | 111,53 giây |
| Trên 300 giây | 3.947 | 192,07 giây |

Các metric này chỉ là đánh giá nghiên cứu trên NIST; chưa phải tiêu chuẩn an
toàn hay chứng nhận cho hệ thống báo cháy.

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
