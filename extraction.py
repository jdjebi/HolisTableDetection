"""
Script réalisant la pipeline complète de l'extraction de données.
"""
from pathlib import Path

import click

from config.path_manager import PathManager
from extract_pdf_scan import extract_scan_tables
from extract_pdf_text import extract_raw_text
from merge_text_scan_data import merge_raw_scan_table
from table_detector import table_detection, merge_pdf_table_images
from utils.utils import makedirs, save


def extract_data(dataset_dir: Path, output_dir: Path, annotate: bool):
    """ Chemins et dossier de sortie """

    pathManager = PathManager(output_dir)

    print(f"Inférence sur le dataset {dataset_dir}")

    """ Création des dossiers de sortie """

    makedirs(output_dir, remove_ok=True)
    makedirs(pathManager.csv_dir)
    makedirs(pathManager.scan_dir)
    makedirs(pathManager.table_detected_dir)

    """ Extraction du texte """

    print("# Extraction du texte")
    raw_text = extract_raw_text(dataset_dir, annotate)
    df_raw_text = save(raw_text, pathManager.raw_text_csv)

    """ Extraction des pages contenant un tableau scanné """

    print("# Extraction des pages contenant une image")
    scan_pages = extract_scan_tables(df_raw_text, pathManager.scan_dir)
    df_scan_pages = save(scan_pages, pathManager.scan_pages_csv)

    """ Détection des tableaux dans la pages extraites """

    print("# Détection des tableaux")
    table_detection(pathManager.scan_dir, pathManager.table_detected_dir)

    print("# Correspondance entre les tableaux détectés et les pdf")
    df_table_images = merge_pdf_table_images(df_scan_pages, pathManager.table_detected_dir)
    save(df_table_images, pathManager.scan_tables_images_csv)

    """ Fusion des datasets de tableau brute et de tableau scanné """

    print("# Fusion des datasets de tableaux brutes et de tableaux scannés")
    dataset = merge_raw_scan_table(df_raw_text, df_table_images)
    save(dataset, pathManager.table_dataset_csv)

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
