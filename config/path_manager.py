from pathlib import Path

from config.constants import CSV_DIR, PATH_TABLE_DETECTED_DIR, PATH_SCAN_DIR, PATH_RAW_TEXT_CSV_FILE, \
    PATH_SCAN_CSV_FILE, PATH_TABLE_DETECTED_CSV_FILE, PATH_DATASET_CSV_FILE


class PathManager:

    def __init__(self, output_dir: Path):
        self.csv_dir = output_dir / CSV_DIR
        self.scan_dir = output_dir / PATH_SCAN_DIR
        self.table_detected_dir = output_dir / PATH_TABLE_DETECTED_DIR

        self.raw_text_csv = output_dir / PATH_RAW_TEXT_CSV_FILE
        self.scan_pages_csv = output_dir / PATH_SCAN_CSV_FILE
        self.scan_tables_images_csv = output_dir / PATH_TABLE_DETECTED_CSV_FILE
        self.table_dataset_csv = output_dir / PATH_DATASET_CSV_FILE
