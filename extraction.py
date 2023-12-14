from pathlib import Path
from typing import Union

import click

from config.constants import ENCODING, CSV_DIR, PDF_RAW_TEXT_CSV_FILE, __DATASET_CSV, __SCAN_TABLES_IMAGES_CSV, \
    __SCAN_TABLES_PAGE_IMAGES_DIR, __SCAN_TABLES_PAGE_IMAGES_CSV, __IMAGE_TABLE_DETECTED_DIR
from merge_text_and_scan_dataset import merge_raw_scan_table
from pdf_scan_table_pages_extraction import scan_page_extraction
from pdf_text_data_extraction import raw_text_extraction
from table_detector import table_detection, merge_pdf_table_images
from utils.utils import save_wth_dataframe, make_and_remove_dir_if_exists, makedirs


def extract_data(dataset_dir: Union[str, Path], output_dir: Union[str, Path], annotate: bool):
    """ Chemins et dossier de sortie """
    raw_text_csv = output_dir / CSV_DIR / PDF_RAW_TEXT_CSV_FILE
    scan_pages_csv = output_dir / CSV_DIR / __SCAN_TABLES_PAGE_IMAGES_CSV
    scan_tables_images_csv = output_dir / CSV_DIR / __SCAN_TABLES_IMAGES_CSV
    table_dataset_csv = output_dir / CSV_DIR / __DATASET_CSV
    csv_dir = output_dir / CSV_DIR
    scan_dir = output_dir / __SCAN_TABLES_PAGE_IMAGES_DIR
    table_detected_dir = output_dir / __IMAGE_TABLE_DETECTED_DIR

    print(f"Inférence sur le dataset {dataset_dir}")

    """ Préparation """

    # Création du dossier de sortie
    make_and_remove_dir_if_exists(output_dir)
    makedirs(csv_dir)
    makedirs(scan_dir)
    makedirs(table_detected_dir)

    """ Extraction du texte """

    print("# Extraction du texte")
    raw_text_data = raw_text_extraction(dataset_dir, annotate)

    # Sauvegarde des données de l'extraction du texte des pdf
    df_raw_text_data = save_wth_dataframe(raw_text_data, raw_text_csv, encoding=ENCODING)

    """ Extraction des pages contenant un tableau scanné """

    print("# Extraction des pages contenant une image")
    scan_pages = scan_page_extraction(df_raw_text_data, scan_dir)

    # Sauvegarde des données du mapping (pdf, page, doc_id, scan_image_path)
    df_scan_pages = save_wth_dataframe(scan_pages, scan_pages_csv, encoding=ENCODING)

    """ Détection des tableaux dans la pages extraites """

    print("# Détection des tableaux")

    # Détection des tableaux
    table_detection(scan_dir, table_detected_dir)

    """ Correspondance entre les tableaux détectés et les pdf """

    print("# Correspondance entre les tableaux détectés et les pdf")
    df_table_images = merge_pdf_table_images(df_scan_pages, table_detected_dir)

    # Sauvegarde de la correspondance
    save_wth_dataframe(df_table_images, scan_tables_images_csv, encoding=ENCODING)

    """ Fusion des datasets de tableau brute et de tableau scanné """

    print("# Fusion des datasets de tableaux brutes et de tableaux scannés")
    dataset = merge_raw_scan_table(df_raw_text_data, df_table_images)

    # Sauvegarde de la correspondance
    save_wth_dataframe(dataset, table_dataset_csv, encoding=ENCODING)

    """ Fin """

    print(f"Extraction terminée! Donnée enregistrée dans {output_dir}")


@click.command()
@click.option("-d", "--dataset-dir", help="Chemin vers le dataset", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option("-o", "--output-dir", help="Chemin vers le dossier de sortie", required=True,
              type=click.Path(resolve_path=True, path_type=Path))
@click.option("-a", "--annotate", help="Indique si les données doivent être annotée", is_flag=True, show_default=True,
              default=False)
def main(dataset_dir: Path, output_dir: Path, annotate: bool):
    extract_data(dataset_dir, output_dir, annotate)


if __name__ == "__main__":
    main()
