from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nist_pipeline.ingest import prepare_nist


if __name__ == "__main__":
    manifest = prepare_nist()
    print(f"Prepared {len(manifest)} NIST experiments.")
    print(manifest.groupby("experiment_type").size().to_string())
    print(f"Labeled experiments: {manifest['alarm_time_sec'].notna().sum()}")
