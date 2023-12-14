from pathlib import Path
from typing import Union

import pytesseract

from config.constants import TESSERACT_EXECUTABLE
from preprocess.text_preprocess import clean_and_flatten_text

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXECUTABLE


def apply_ocr(image_path: Union[str, Path]) -> str:
    text = pytesseract.image_to_string(image_path)
    text = clean_and_flatten_text(text)
    return text
