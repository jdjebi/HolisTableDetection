"""
Ce script gère l'extraction des pages des pdf qui contiennent des tableaux scannés.

Il utilise les résultats de l'extraction faite avec pdf_text_data_extraction.py
pour cibler les pages des pdf concerné.
"""
from pathlib import Path
from typing import Union, List, Dict

import click
import pandas as pd
from fitz import fitz
from tqdm import tqdm

from config.constants import CSV_DIR, \
    PDF_RAW_TEXT_CSV_FILE, __SCAN_TABLES_PAGE_IMAGES_DIR, __SCAN_TABLES_PAGE_IMAGES_CSV, ENCODING
from utils.utils import make_and_remove_dir_if_exists, save_wth_dataframe


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
                pix = page.get_pixmap(dpi=300)
                output_file = scan_output_dir / f"{doc_id}__{pdf_path.stem[:30]}-page-{num_page}.jpg"
                pix.save(output_file)

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "page": num_page,
                    "image": output_file.stem[:30]
                })

                data.append({
                    "path": str(pdf_path),
                    "page": num_page,
                    "scan_path": str(output_file),
                    "doc_id": doc_id
                })

    return data


@click.command()
@click.option("-o", "--output-dir", help="Chemin vers le dossier de sortie", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
def main(output_dir: Path):
    """ Dossier de sortie """

    raw_text_csv = output_dir / CSV_DIR / PDF_RAW_TEXT_CSV_FILE
    scan_tables_images_csv = output_dir / CSV_DIR / __SCAN_TABLES_PAGE_IMAGES_CSV
    scan_tables_images_dir = output_dir / __SCAN_TABLES_PAGE_IMAGES_DIR

    # Vérifiez que le fichier csv de l'extraction des textes bruts des pdf existe
    if not raw_text_csv.exists():
        raise ValueError(f"Fichier de l'extraction des textes bruts '{PDF_RAW_TEXT_CSV_FILE}' inexistant.\n"
                         f"Veuillez d'abord faire l'extraction des textes en exécutant le script "
                         f"'pdf_text_data_extraction.py', avec comme dossier de sortie {output_dir}")

    # On supprime le dossier de sortie s'il existe pour ne pas mélanger les sorties
    make_and_remove_dir_if_exists(scan_tables_images_dir)

    print(f"PDF image data Extraction starting from : {raw_text_csv}")

    """ Préparation des données """

    # Chargement des données
    df_raw_text_data = pd.read_csv(raw_text_csv, sep=";", index_col=0)

    # Extraction des pages
    data = scan_page_extraction(df_raw_text_data, scan_tables_images_dir)

    print(f"Extraction finished!")
    print(f"CSV mapping saved at : {scan_tables_images_csv}")

    # Sauvegarde des données
    save_wth_dataframe(data, scan_tables_images_csv, encoding=ENCODING)


if __name__ == "__main__":
    main()
