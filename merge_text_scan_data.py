"""
Ce script est responsable de la fusion des datasets des tableaux natifs et des tableaux scannés
"""
from pathlib import Path

import click
import pandas as pd

from config.constants import ENCODING, PATH_TABLE_DETECTED_CSV_FILE, PATH_RAW_TEXT_CSV_FILE, PATH_DATASET_CSV_FILE
from utils.utils import save_wth_dataframe


def merge_raw_scan_table(df_raw_txt: pd.DataFrame, df_table_images: pd.DataFrame) -> pd.DataFrame:
    df_table_images = df_table_images[["doc_id", "text"]]

    merged_df = pd.merge(df_raw_txt, df_table_images, on='doc_id', how='outer')
    merged_df["text"] = merged_df[['text_x', 'text_y']].agg(lambda x: ' '.join(x.dropna()), axis=1)

    cols = ["path", "page", "is_scan", "text"]
    if "interest_table" in df_raw_txt.columns:
        cols.append("interest_table")

    dataset = merged_df[cols]

    return dataset


@click.command()
@click.option("-o", "--output-dir", help="Chemin vers le dossier de sortie", required=True,
              type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
def main(output_dir: Path):

    raw_text_csv = output_dir / PATH_RAW_TEXT_CSV_FILE
    scan_tables_images_csv = output_dir / PATH_TABLE_DETECTED_CSV_FILE
    table_dataset_csv = output_dir / PATH_DATASET_CSV_FILE

    print(f"Merge text dataset and scan dataset to create final dataset")

    df_raw_txt = pd.read_csv(raw_text_csv, sep=";", index_col=0)
    df_table_images = pd.read_csv(scan_tables_images_csv, sep=";", index_col=0)

    dataset = merge_raw_scan_table(df_raw_txt, df_table_images)

    # Sauvegarde de la correspondance
    save_wth_dataframe(dataset, table_dataset_csv, encoding=ENCODING)

    print(f"Saved dataset in : {table_dataset_csv}")


if __name__ == "__main__":
    main()
