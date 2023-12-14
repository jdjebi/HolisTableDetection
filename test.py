import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from preprocess.text_preprocess import clean_and_flatten_text
from table_transformer.src.inference import TableExtractionPipeline

import pytesseract

# Chemins
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

tables_output_path = Path(r"datasets/tmp/tables")
im_page_with_table_path = Path(r"datasets/tmp/images/6f92ff8f__CAI2.J.825-02  MARSEILLE - POR-page-28.jpg")

filename = im_page_with_table_path.stem
output_path = im_page_with_table_path.parent

pipe = TableExtractionPipeline(
    det_device="cpu",
    str_device="cpu",
    det_model_path="table_transformer/model/pubtables1m_detection_detr_r18.pth",
    str_model_path="table_transformer/model/TATR-v1.1-All-msft.pth",
    det_config_path="table_transformer/src/detection_config.json",
    str_config_path="table_transformer/src/structure_config.json",
)

_im_page_with_table = Image.open(im_page_with_table_path)

# Détection de tableau

extracted_tables = pipe.detect(
    img=_im_page_with_table,
    tokens=[],
    out_objects=True,
    crop_padding=1
)

# padding_x = 80
# padding_y = 100

padding_x = 0
padding_y = 0

for i, obj in enumerate(extracted_tables["objects"]):
    bbox = obj["bbox"]
    bbox = tuple([int(coord) for coord in bbox])
    bbox = (bbox[0] - padding_x), (bbox[1] - padding_y),  (bbox[2] + padding_x), (bbox[3] + padding_y)
    cropped_image = _im_page_with_table.crop(bbox)
    cropped_image.save(tables_output_path / f"{filename}_{i}_table.jpg")

    # Reconnaissance

    im_ocr_df: pd.DataFrame = pytesseract.image_to_data(cropped_image, lang='eng+fra', output_type=pytesseract.Output.DATAFRAME)

    tokens = []

    im_ocr_df.to_csv("test_df.csv")

    for index, row in im_ocr_df.iterrows():
        left = row["left"]
        top = row["top"]
        right = left + row["width"]
        bottom = top + row["height"]

        text = str(row["text"])

        text = text.replace(",", "")

        if text.strip() == "":
            text = " "

        if row["conf"] > 0 or text != "nan":
            tokens.append({
                "bbox": [left, top, right, bottom],
                "text": text,
                "block_num": row["block_num"],
                "line_num": row["line_num"],
                "span_num": row["par_num"]
            })

    with open(tables_output_path / f"{filename}_word.json", mode="w") as file:
        json.dump(tokens, file, indent=4)

    extracted_tables = pipe.recognize(_im_page_with_table, tokens, out_objects=True, out_cells=True, out_html=True, out_csv=True)

    objects = extracted_tables['objects']
    cells = extracted_tables['cells']
    csv = extracted_tables['csv']
    html = extracted_tables['html']
    # print(extracted_tables)

    with open(tables_output_path / f"{filename}_{i}_table.csv", mode="w") as file:
        for line in csv:
            file.write(line)

    with open(tables_output_path / f"{filename}_test_cells_{i}.json", mode="w") as file:
        json.dump(cells, file, indent=4)

    with open(tables_output_path / f"{filename}_test_objects_{i}.json", mode="w") as file:
        json.dump(objects, file, indent=4)
