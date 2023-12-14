"""
Ce script est responsable de la fusion des datasets des tableaux natifs et des tableaux scanné
"""
from pathlib import Path

import click
import pandas as pd

from config.constants import RAW_TXT_CSV, SCAN_TABLES_IMAGES_CSV, ENCODING, DATASET_CSV, CSV_DIR, __DATASET_CSV, \
    PDF_RAW_TEXT_CSV_FILE, __SCAN_TABLES_IMAGES_CSV
from utils.utils import save_wth_dataframe


def merge_raw_scan_table(df_raw_txt: pd.DataFrame, df_table_images: pd.DataFrame) -> pd.DataFrame:
    df_table_images = df_table_images[["doc_id", "text"]]

    merged_df = pd.merge(df_raw_txt, df_table_images, on='doc_id', how='outer')
    merged_df["text"] = merged_df[['text_x', 'text_y']].agg(lambda x: ' '.join(x.dropna()), axis=1)

    dataset = merged_df[["path", "page", "is_scan", "text"]]

    return dataset


@click.command()
@click.option("-o", "--output-dir", help="Chemin vers le dossier de sortie", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
def main(output_dir: Path):

    """ Chemins """

    raw_text_csv = output_dir / CSV_DIR / PDF_RAW_TEXT_CSV_FILE
    scan_tables_images_csv = output_dir / CSV_DIR / __SCAN_TABLES_IMAGES_CSV
    table_dataset_csv = output_dir / CSV_DIR / __DATASET_CSV

    print(f"Merge text dataset and scan dataset to create final dataset")

    df_raw_txt = pd.read_csv(raw_text_csv, sep=";", index_col=0)
    df_table_images = pd.read_csv(scan_tables_images_csv, sep=";", index_col=0)

    dataset = merge_raw_scan_table(df_raw_txt, df_table_images)

    # Sauvegarde de la correspondance
    save_wth_dataframe(dataset, table_dataset_csv, encoding=ENCODING)

    print(f"Saved dataset in : {table_dataset_csv}")


if __name__ == "__main__":
    main()
