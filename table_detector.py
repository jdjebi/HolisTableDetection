"""
Ce script permet de détecter et d'extraire les images des tableaux de données à partir des images contenu dans le dossier
datasets/scan_table_page_images. Les images des tableaux extraites sont enregistrées dans le dossier datasets/image_table_detected

Une fois l'exécution terminée (donc les tableaux extraits). Les tableaux sont associés à leur pdf et page correspondante
à partir du fichier datasets/csv/scan_table_page_images.csv.
"""
import os
import shutil

import pandas as pd

from config.constants import INFERENCE_SCRIPT, SCAN_TABLES_IMAGES_DIR, DETECTION_CONFIG_PATH, MODEL_PATH, IMAGE_TABLE_DETECTED_DIR, \
    SCAN_TABLE_PAGE_IMAGES_CSV, SCAN_TABLES_IMAGES_CSV, ENCODING
from utils.ocr import apply_ocr
from utils.utils import makedirs, save_wth_dataframe

command_parts = {
    "python": str(INFERENCE_SCRIPT),
    "--image_dir": str(SCAN_TABLES_IMAGES_DIR),
    "--out_dir": str(IMAGE_TABLE_DETECTED_DIR),
    "--mode": "detect",
    "--detection_config_path": str(DETECTION_CONFIG_PATH),
    "--detection_device": "cpu",
    "--structure_device": "cpu",
    "--detection_model_path": str(MODEL_PATH),
    "--crops": "",
    "--crop_padding": 5,
    "--visualize": ""
}


def main():

    """ Détection des tableaux dans les pages scannées """

    # On supprime le dossier de sortie s'il existe pour ne pas mélanger les sorties
    if IMAGE_TABLE_DETECTED_DIR.exists():
        shutil.rmtree(IMAGE_TABLE_DETECTED_DIR)

    makedirs(IMAGE_TABLE_DETECTED_DIR)

    print(f"Detect table from images dataset : {SCAN_TABLES_IMAGES_DIR}")

    # Construction de la commande pour détecter les tableaux
    cmd_parts = " ".join([f"{action} {value}" for action, value in command_parts.items()])
    os.system(cmd_parts)

    print(f"Detection finished! Result saved in : {IMAGE_TABLE_DETECTED_DIR}")

    """ Fusion entre les pages scannées est les tableaux prédit """

    print("Correspondence between table images and PDF pages")

    # Dataframe des pages scannée
    df_scan_table_pages = pd.read_csv(SCAN_TABLE_PAGE_IMAGES_CSV, sep=";", index_col=0)

    # Récupération des id de page scanné et du chemin des images des tableaux
    id_im_table_path = []

    for path in list(IMAGE_TABLE_DETECTED_DIR.glob("*")):

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

    merged_df = pd.merge(df_scan_table_pages, df_images_table, on='doc_id')

    # Sauvegarde de la correspondance
    save_wth_dataframe(merged_df, SCAN_TABLES_IMAGES_CSV, encoding=ENCODING)

    print(f"Merged scans and tables saved at : {SCAN_TABLES_IMAGES_CSV}")


if __name__ == "__main__":
    main()
