"""
Ce script gère l'extraction des pages des pdf qui contiennent des tableaux scannés.

Il utilise les résultats de l'extraction faite avec pdf_text_data_extraction.py
pour cibler les pages des pdf concerné.
"""
import shutil
from pathlib import Path

import pandas as pd
from fitz import fitz
from tqdm import tqdm

from config.constants import RAW_TXT_CSV, SCAN_TABLE_PAGE_IMAGES_CSV, ENCODING, SCAN_TABLES_IMAGES_DIR
from utils.utils import makedirs, generate_unique_id, save_wth_dataframe

""" Constantes """


def main():
    print(f"PDF image data Extraction starting from : {RAW_TXT_CSV}")

    """ Dossier de sortie """

    # On supprime le dossier de sortie s'il existe pour ne pas mélanger les sorties
    if SCAN_TABLES_IMAGES_DIR.exists():
        shutil.rmtree(SCAN_TABLES_IMAGES_DIR)

    makedirs(SCAN_TABLES_IMAGES_DIR)

    """ Préparation des données """

    # Chargement des données
    df = pd.read_csv(RAW_TXT_CSV, sep=";", index_col=0)

    # Sélection des pages scannées
    df_scan_pages = df[df.is_scan == True]

    # Récupération des pdf et des numéros de page """
    pdf_files = [Path(path) for path in df_scan_pages.path.values]
    docs_id = df_scan_pages.doc_id.values
    scan_pages_id = df_scan_pages.page.values

    pdf_pages = zip(docs_id, pdf_files, scan_pages_id)

    data = []

    """ Extraction """

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
                output_file = SCAN_TABLES_IMAGES_DIR / f"{doc_id}__{pdf_path.stem}-page-{num_page}.jpg"
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

    print(f"Extraction finished! Images saved at : {SCAN_TABLE_PAGE_IMAGES_CSV}")
    print(f"CSV mapping saved at : {SCAN_TABLE_PAGE_IMAGES_CSV}")

    # Sauvegarde des données
    save_wth_dataframe(data, SCAN_TABLE_PAGE_IMAGES_CSV, encoding=ENCODING)


if __name__ == "__main__":
    main()
