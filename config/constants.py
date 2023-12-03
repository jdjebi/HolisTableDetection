from pathlib import Path

ENCODING = "utf-8-sig"

DATASET_DIR = Path("datasets") / "csv"
DATASET_SAVE_PATH = (DATASET_DIR / "pdf_raw_text.csv").absolute().resolve()
IMAGE_TABLE_DETECTED_DIR = Path("datasets/image_table_detected").absolute().resolve()
TABLE_DETECTION_OUTPUT_DIR = Path("datasets/image_table_detected").absolute().resolve()
SCAN_TABLES_IMAGES_DIR = Path("datasets/scan_table_page_images").absolute().resolve()

RAW_TXT_CSV = "datasets/csv/pdf_raw_text.csv"
SCAN_TABLE_PAGE_IMAGES_CSV = Path("datasets/csv/scan_table_page_images.csv").absolute().resolve()
SCAN_TABLES_IMAGES_CSV = Path("datasets/csv/scan_table_images.csv").absolute().resolve()
OUTPUT_CSV = Path("datasets/csv/scan_table_page_images.csv").absolute().resolve()
DATASET_CSV = Path("datasets/csv/table_dataset.csv").absolute().resolve()

INFERENCE_SCRIPT = Path("table-transformer/src/inference.py").absolute().resolve()
MODEL_PATH = Path("table-transformer/model/pubtables1m_detection_detr_r18.pth").absolute().resolve()
DETECTION_CONFIG_PATH = Path("table-transformer/src/detection_config.json").absolute().resolve()

TESSERACT_EXECUTABLE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
