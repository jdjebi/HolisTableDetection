"""
Ce script permet de détecter et d'extraire les images des tableaux de données à partir des images contenu dans le dossier
datasets/scan_table_page_images. Les images des tableaux extraites sont enregistrées dans le dossier datasets/image_table_detected

Une fois l'exécution terminée (donc les tableaux extraits). Les tableaux sont associés à leur pdf et page correspondante
à partir du fichier datasets/csv/scan_table_page_images.csv.
"""
import os
import sys
from pathlib import Path
from typing import Union

import click
import pandas as pd
from tqdm import tqdm

from config.constants import INFERENCE_SCRIPT, DETECTION_CONFIG_PATH, MODEL_PATH, \
    ENCODING, PATH_SCAN_DIR, PATH_SCAN_CSV_FILE, \
    PATH_TABLE_DETECTED_DIR, PATH_TABLE_DETECTED_CSV_FILE
from utils.ocr import apply_ocr
from utils.utils import save_wth_dataframe, make_and_remove_dir_if_exists

# Chemin vers l'exécutable python en cours
PYTHON_EXEC = sys.executable


def table_detection(scan_dir: Union[str, Path], output_dir: Union[str, Path]):
    command_parts = {
        PYTHON_EXEC: str(INFERENCE_SCRIPT),
        "--image_dir": str(scan_dir),
        "--out_dir": str(output_dir),
        "--mode": "detect",
        "--detection_config_path": str(DETECTION_CONFIG_PATH),
        "--detection_device": "cpu",
        "--structure_device": "cpu",
        "--detection_model_path": str(MODEL_PATH),
        "--verbose": "",
        "--crops": "",
        "--crop_padding": 5,
    }

    # Construction de la commande pour détecter les tableaux
    cmd_parts = " ".join([f"{action} {value}" for action, value in command_parts.items()])
    os.system(cmd_parts)


def merge_pdf_table_images(df_scan_table_pages: pd.DataFrame, image_table_detected: Union[str, Path]) -> pd.DataFrame:
    # Récupération des id de page scanné et du chemin des images des tableaux
    id_im_table_path = []

    images_list = list(image_table_detected.glob("*.jpg"))

    with tqdm(total=len(images_list), desc="OCR", unit="image") as progress_bar:
        for path in images_list:
            # On récupère uniquement les .jpg

            progress_bar.set_postfix({
                "image": path.stem[:30]
            })

            filename = path.stem
            doc_id, _ = filename.split("__")

            text = apply_ocr(str(path))

            id_im_table_path.append({
                "doc_id": doc_id,
                "im_table_path": str(path),
                "text": text
            })

            progress_bar.update(1)

    # Création d'un DataFrame des id de scan et des images des tableaux
    df_images_table = pd.DataFrame(id_im_table_path)

    merged_df = pd.merge(df_scan_table_pages, df_images_table, on='doc_id', how="outer")

    return merged_df


@click.command()
@click.option("-o", "--output-dir", help="Chemin vers le dossier de sortie", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
def main(output_dir: Path):
    """ Sortie """

    # Vérification de la présence du dossier de l'extraction des images
    scan_tables_page_images_dir = output_dir / PATH_SCAN_DIR
    scan_tables_page_images_csv = output_dir / PATH_SCAN_CSV_FILE
    scan_tables_images_csv = output_dir / PATH_TABLE_DETECTED_CSV_FILE
    images_table_detected_dir = output_dir / PATH_TABLE_DETECTED_DIR

    # On supprime le dossier de sortie s'il existe pour ne pas mélanger les sorties
    make_and_remove_dir_if_exists(images_table_detected_dir)

    print(f"Detect table from images dataset : {scan_tables_page_images_dir}")

    """ Détection des tableaux dans les pages scannées """

    # Construction de la commande pour détecter les tableaux
    table_detection(scan_tables_page_images_dir, images_table_detected_dir)

    print(f"Detection finished! Result saved in : {images_table_detected_dir}")

    """ Fusion entre les pages scannées est les tableaux prédit """

    print("Correspondence between table images and PDF pages")

    # Dataframe des pages scannée
    df_scan_table_pages = pd.read_csv(scan_tables_page_images_csv, sep=";", index_col=0)

    df_table_images = merge_pdf_table_images(df_scan_table_pages, images_table_detected_dir)

    # Sauvegarde de la correspondance
    save_wth_dataframe(df_table_images, scan_tables_images_csv, encoding=ENCODING)

    print(f"Merged scans and tables saved at : {scan_tables_images_csv}")


if __name__ == "__main__":
    main()
