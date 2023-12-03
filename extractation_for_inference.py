"""
Ce script permet de faire l'inférence sur le dataset placé dans DOCUMENTS_PATH.
"""

import os
import shutil
from pathlib import Path
from typing import Union, List, Dict

import pandas as pd
from fitz import fitz
from tqdm import tqdm

from config.constants import ENCODING, MODEL_PATH, DETECTION_CONFIG_PATH, INFERENCE_SCRIPT
from utils.extraction_toolkit import extract_pdf_data
from utils.ocr import apply_ocr
from utils.utils import get_pdf_files, makedirs, save_wth_dataframe

DOCUMENTS_PATH = r"C:\Users\jeanm\Downloads\DOC DATA CHALLENGE\DOC DATA CHELLENGE\SPARE_DOCUMENT"

TEST_OUTPUT_DIR = r".\test_datasets2"
TEST_OUTPUT_DIR = Path(TEST_OUTPUT_DIR).absolute().resolve()


def raw_text_extraction(dataset_dir: Union[str, Path]) -> List[Dict]:
    # Chemins
    pdf_files: List[Path] = get_pdf_files(dataset_dir)

    # Extraction de données
    data: List[Dict] = extract_pdf_data(pdf_files, annotate=False)

    return data


def scan_page_extraction(df_raw_text_data: pd.DataFrame, scan_output_dir: Union[str, Path]) -> List[Dict]:
    # Sélection des pages scannées
    df_scan_pages = df_raw_text_data[df_raw_text_data.is_scan == True]

    # Récupération des pdf et des numéros de page """
    pdf_files = [Path(path) for path in df_scan_pages.path.values]
    docs_id = df_scan_pages.doc_id.values
    scan_pages_id = df_scan_pages.page.values

    # Compression des données
    pdf_pages = zip(docs_id, pdf_files, scan_pages_id)

    # Extraction
    data = []

    with tqdm(total=len(pdf_files), unit="pdf", desc=f"Extraction") as progress_bar:
        for doc_id, pdf_path, num_page in pdf_pages:
            with fitz.open(pdf_path) as pdf_document:

                # Sélection de la bonne page
                num_page = int(num_page)
                page = pdf_document[num_page]

                # Suppression des annotations
                for annot in page.annots():
                    page.delete_annot(annot)

                # Sauvegarde de la page sous forme d'images
                pix = page.get_pixmap()
                output_file = scan_output_dir / f"{doc_id}__{pdf_path.stem[:30]}-page-{num_page}.jpg"
                pix.save(output_file)

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "page": num_page,
                    "image": output_file.stem
                })

                data.append({
                    "path": str(pdf_path),
                    "page": num_page,
                    "scan_path": str(output_file),
                    "doc_id": doc_id
                })

    return data


def table_detection(scan_dir: Union[str, Path], output_dir: Union[str, Path]):
    command_parts = {
        "python": str(INFERENCE_SCRIPT),
        "--image_dir": str(scan_dir),
        "--out_dir": str(output_dir),
        "--mode": "detect",
        "--detection_config_path": str(DETECTION_CONFIG_PATH),
        "--detection_device": "cpu",
        "--structure_device": "cpu",
        "--detection_model_path": str(MODEL_PATH),
        "--crops": "",
        "--crop_padding": 5,
        "--visualize": ""
    }

    # Construction de la commande pour détecter les tableaux
    cmd_parts = " ".join([f"{action} {value}" for action, value in command_parts.items()])
    os.system(cmd_parts)


def merge_pdf_table_images(df_scan_table_pages: pd.DataFrame, image_table_detected: Union[str, Path]) -> pd.DataFrame:
    # Récupération des id de page scanné et du chemin des images des tableaux
    id_im_table_path = []

    for path in list(image_table_detected.glob("*")):
        # On récupère uniquement les .jpg

        if path.suffix != ".jpg":
            continue

        filename = path.stem
        doc_id, _ = filename.split("__")

        id_im_table_path.append({
            "doc_id": doc_id,
            "im_table_path": str(path),
            "text": apply_ocr(str(path))
        })

        # Création d'un DataFrame des id de scan et des images des tableaux
        df_images_table = pd.DataFrame(id_im_table_path)

        merged_df = pd.merge(df_scan_table_pages, df_images_table, on='doc_id', how="outer")

        return merged_df


def merge_raw_scan_table(df_raw_txt: pd.DataFrame, df_table_images: pd.DataFrame) -> pd.DataFrame:
    df_table_images = df_table_images[["doc_id", "text"]]

    merged_df = pd.merge(df_raw_txt, df_table_images, on='doc_id', how='outer')
    merged_df["text"] = merged_df[['text_x', 'text_y']].agg(lambda x: ' '.join(x.dropna()), axis=1)

    dataset = merged_df[["path", "page", "is_scan", "text"]]

    return dataset


def extract_data(dataset_dir: Union[str, Path], output_dir: Union[str, Path]):

    print(f"Inférence sur le dataset {dataset_dir}")

    """ Préparation """

    # Création du dossier de sortie
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Création des sous dossier de sortie
    makedirs(output_dir)
    makedirs(output_dir / "csv")
    makedirs(output_dir / "image_table_detected")
    makedirs(output_dir / "scan_table_page_images")

    """ Extraction du texte """

    print("# Extraction du texte")
    raw_text_data = raw_text_extraction(dataset_dir)

    # Sauvegarde des données de l'extraction du texte des pdf
    df_raw_text_data = save_wth_dataframe(raw_text_data, output_dir / "csv" / "pdf_raw_text.csv", encoding=ENCODING)

    """ Extraction des pages contenant un tableau scanné """

    print("# Extraction des pages contenant une image")
    scan_table_pages_images = scan_page_extraction(df_raw_text_data, output_dir / "scan_table_page_images")

    # Sauvegarde des données du mapping (pdf, page, doc_id, scan_image_path)
    df_scan_table_pages = save_wth_dataframe(scan_table_pages_images, output_dir / "csv" / "scan_table_page_images.csv",
                                             encoding=ENCODING)

    """ Détection des tableaux dans la pages extraites """

    print("# Détection des tableaux")
    image_table_detected = output_dir / "scan_table_page_images"
    table_detected_dir = output_dir / "image_table_detected"
    table_detection(image_table_detected, table_detected_dir)

    # Détection des tableaux
    table_detection(image_table_detected, table_detected_dir)

    """ Correspondance entre les tableaux détectés et les pdf """

    print("# Correspondance entre les tableaux détectés et les pdf")
    df_table_images = merge_pdf_table_images(df_scan_table_pages, image_table_detected)

    # Sauvegarde de la correspondance
    save_wth_dataframe(df_table_images, output_dir / "csv" / "scan_table_images.csv", encoding=ENCODING)

    """ Fusion des datasets de tableau brute et de tableau scanné """

    print("# Fusion des datasets de tableaux brutes et de tableaux scannés")
    dataset = merge_raw_scan_table(df_raw_text_data, df_table_images)

    # Sauvegarde de la correspondance
    save_wth_dataframe(dataset, output_dir / "csv" / "table_dataset.csv", encoding=ENCODING)

    """ Fin """

    print(f"Extraction terminées! Donnée enregistrée dans {output_dir}")


def main(dataset_dir: Union[str, Path], output_dir: Union[str, Path]):

    # Extraction des données
    extract_data(dataset_dir, output_dir)


if __name__ == "__main__":
    main(DOCUMENTS_PATH, TEST_OUTPUT_DIR)
