from pathlib import Path

# Encodage de sortie
ENCODING = "utf-8-sig"

# Table transformer
INFERENCE_SCRIPT = Path("table_transformer/src/inference.py").absolute().resolve()
MODEL_PATH = Path("table_transformer/model/pubtables1m_detection_detr_r18.pth").absolute().resolve()
DETECTION_CONFIG_PATH = Path("table_transformer/src/detection_config.json").absolute().resolve()

# OCR
TESSERACT_EXECUTABLE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Sous-dossier d'un dossier de sortie
CSV_DIR = "csv"
__SCAN_TABLES_PAGE_IMAGES_DIR = "scan_table_page_images"
__IMAGE_TABLE_DETECTED_DIR = "image_table_detected"

PDF_RAW_TEXT_CSV_FILE = "pdf_raw_text.csv"
__SCAN_TABLES_PAGE_IMAGES_CSV = "scan_table_page_images.csv"
__SCAN_TABLES_IMAGES_CSV = "scan_table_images.csv"
__DATASET_CSV = "table_dataset.csv"


# Dossier
PATH_CSV_DIR = Path(CSV_DIR)
PATH_SCAN_DIR = Path(__SCAN_TABLES_PAGE_IMAGES_DIR)
PATH_TABLE_DETECTED_DIR = Path(__IMAGE_TABLE_DETECTED_DIR)

# CSV
PATH_RAW_TEXT_CSV_FILE = PATH_CSV_DIR / PDF_RAW_TEXT_CSV_FILE
PATH_SCAN_CSV_FILE = PATH_CSV_DIR / __SCAN_TABLES_PAGE_IMAGES_CSV
PATH_TABLE_DETECTED_CSV_FILE = PATH_CSV_DIR / __SCAN_TABLES_IMAGES_CSV
PATH_DATASET_CSV_FILE = PATH_CSV_DIR / __DATASET_CSV
