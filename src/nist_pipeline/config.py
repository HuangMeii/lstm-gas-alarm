from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NIST_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nist"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "nist"
MODEL_DIR = PROJECT_ROOT / "models" / "nist"
REPORT_DIR = PROJECT_ROOT / "reports" / "nist"
SUMMARY_WORKBOOK = NIST_RAW_DIR / "archives" / "summary_nistsmokebox_roomresults.xlsx"
