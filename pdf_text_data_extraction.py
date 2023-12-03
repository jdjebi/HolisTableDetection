"""
Script permettant d'extraire les données des tableaux des pdf.

Gère la labellisation des données de sorte que les tableaux annotés aient la classe positive (1) et ceux non-annotés
la classe négative.

Dans le cas des pages ayant la classe négative, toutes les données de la page sont conservés.
Dans le cas des tableaux scannés, les pages concernés sont conservées et l'extraction (OCR sera faite avec un autre
script en extrayant uniquement les pages ayant un tableau scanné du dataset généré par ce script.
"""

from pathlib import Path
from typing import List, Dict

from config.constants import DATASET_DIR, ENCODING, DATASET_SAVE_PATH
from utils.extraction_toolkit import extract_pdf_data
from utils.utils import get_pdf_files, makedirs, save_wth_dataframe

""" Dossier des pdf """

# Chemin vers le jeu de données
SRC_PATH = r"C:\Users\jeanm\Downloads\DOC DATA CHALLENGE\DOC DATA CHELLENGE\SPARE_EXPLICATIONS"


""" Dossier du dataset """

makedirs(DATASET_DIR)


def main(src_path):

    print(f"PDF raw data Extraction starting from : {src_path}")

    # Chemins
    pdf_files: List[Path] = get_pdf_files(src_path)

    # Extraction de données
    data: List[Dict] = extract_pdf_data(pdf_files)

    # Sauvegarde des données
    save_wth_dataframe(data, DATASET_SAVE_PATH, encoding=ENCODING)

    print(f"Extraction finished! Dataset save at : {DATASET_SAVE_PATH}")


if __name__ == "__main__":
    main(SRC_PATH)
