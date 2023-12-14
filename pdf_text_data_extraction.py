"""
Script permettant d'extraire les données des tableaux des pdf.

Gère la labellisation des données de sorte que les tableaux annotés aient la classe positive (1) et ceux non-annotés
la classe négative.

Dans le cas des pages ayant la classe négative, toutes les données de la page sont conservés.
Dans le cas des tableaux scannés, les pages concernés sont conservées et l'extraction (OCR sera faite avec un autre
script en extrayant uniquement les pages ayant un tableau scanné du dataset généré par ce script.
"""

from pathlib import Path
from typing import List, Dict, Union

import click

from config.constants import ENCODING, CSV_DIR, PDF_RAW_TEXT_CSV_FILE
from utils.extraction_toolkit import extract_pdf_data
from utils.utils import get_pdf_files, makedirs, save_wth_dataframe


def raw_text_extraction(dataset_dir: Union[str, Path], annotate: bool = False) -> List[Dict]:
    # Chemins
    pdf_files: List[Path] = get_pdf_files(dataset_dir)

    # Extraction de données
    data: List[Dict] = extract_pdf_data(pdf_files, annotate=annotate)

    return data


@click.command()
@click.option("-d", "--dataset-dir", help="Chemin vers le dataset", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
@click.option("-o", "--output-dir", help="Chemin vers le dossier de sortie", required=True,
              type=click.Path(resolve_path=True, path_type=Path))
def main(dataset_dir: Path, output_dir: Path):
    # Chemin et dossier de sortie
    makedirs(output_dir / CSV_DIR)
    output_dir_csv = output_dir / CSV_DIR / PDF_RAW_TEXT_CSV_FILE

    print(f"PDF raw data Extraction starting from : {dataset_dir}")

    # Extraction des données textuelles des pdf
    data: List[Dict] = raw_text_extraction(dataset_dir)

    # Sauvegarde des données
    save_wth_dataframe(data, output_dir_csv, encoding=ENCODING)

    print(f"Extraction finished! Dataset save at : {output_dir}")


if __name__ == "__main__":
    main()
