"""
Ce script est responsable de la fusion des datasets des tableaux natifs et des tableaux scanné
"""
import pandas as pd

from config.constants import RAW_TXT_CSV, SCAN_TABLES_IMAGES_CSV, ENCODING, DATASET_CSV
from utils.utils import save_wth_dataframe


def main():

    print(f"Merge text dataset and scan dataset to create final dataset")

    df_raw_txt = pd.read_csv(RAW_TXT_CSV, sep=";", index_col=0)
    df_table_images = pd.read_csv(SCAN_TABLES_IMAGES_CSV, sep=";", index_col=0)

    df_table_images = df_table_images[["doc_id", "text"]]

    merged_df = pd.merge(df_raw_txt, df_table_images, on='doc_id', how='outer')
    merged_df["text"] = merged_df[['text_x', 'text_y']].agg(lambda x: ' '.join(x.dropna()), axis=1)

    dataset = merged_df[["path", "page", "is_scan", "text", "interest_table"]]

    # Sauvegarde de la correspondance
    save_wth_dataframe(dataset, DATASET_CSV, encoding=ENCODING)

    print(f"Saved dataset in : {DATASET_CSV}")


if __name__ == "__main__":
    main()
