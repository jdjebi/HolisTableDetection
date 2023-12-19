from pathlib import Path

import click
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from config.constants import ENCODING
from ml.constants import RAMDOM_SATE, TEST_SIZE, MODELS_DIR, PLOT_DIR
from ml.models import train_lr_model, train_rf_model, train_svm_model
from ml.plot import plot_dist_interest_table, plot_items_scan_interesting, plot_items_raw_interesting, \
    plot_items_scan_not_interesting, plot_items_raw_not_interesting
from ml.preprocess_func import cleaning_pdf_interest
from utils.utils import makedirs


# TODO: Dans l'extraction, trouver un moyen pour préciser qu'une page contient à la fois du texte et des images (
#  text_only: False)
def train(train_data_path: Path, output_dir: Path):
    output_models_dir = output_dir / MODELS_DIR
    output_plots_dir = output_dir / PLOT_DIR

    makedirs(output_models_dir, exist_ok=True)
    makedirs(output_plots_dir, exist_ok=True)

    print("# Entraînement")

    # Chargement des données
    train_df = pd.read_csv(train_data_path, delimiter=';', encoding=ENCODING)
    train_set = train_df[['text', 'interest_table']]

    # Distribution de données par tableau d'intérêt
    plot_dist_interest_table(train_set)

    # Nettoyage des données
    print(f"Nettoyage des données...")
    train_set_cleaned = cleaning_pdf_interest(train_set, 'text')

    # Division des données
    X, y = train_set_cleaned['text'], train_set_cleaned['interest_table']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RAMDOM_SATE, stratify=y)

    # Vectorisation
    tfidf_vector = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2)
    X_train, X_test = tfidf_vector.fit_transform(X_train), tfidf_vector.transform(X_test)

    features_names = tfidf_vector.get_feature_names_out()
    print(f"Quelques features : {features_names[:5]}")

    # Entraînement
    print(f"\nEntraînement des modèles...")

    print("## LogisticRegression")
    lr = train_lr_model(X_train, y_train, X_test, y_test, features_names, output_dir)

    print("## RandomForestClassifier")
    rf = train_rf_model(X_train, y_train, X_test, y_test, features_names, output_dir)

    print("## LinearSVC")
    svm = train_svm_model(X_train, y_train, X_test, y_test, features_names, output_dir)

    # Sauvegarde
    print(f"\nSauvegarde du vectoriseur et des modèles...")
    joblib.dump(tfidf_vector, output_models_dir / "tfidf_vector.pkl")
    joblib.dump(lr, output_models_dir / "lr_model.pkl")
    joblib.dump(rf, output_models_dir / "rf_model.pkl")
    joblib.dump(svm, output_models_dir / "svm_model.pkl")

    print(f"Sauvegarde terminée! Dossier de sauvegarde {output_models_dir}")


def test(test_data_path: Path, model_path: Path, vectorizer_path: Path, output_dir: Path):
    print("# Test")

    output_samples_dir = output_dir / "samples"

    makedirs(output_samples_dir, remove_ok=True)

    # Chargement du modèle et du vectoriseur
    print("Chargement du modèle...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Chargement des données
    print("Chargement des données...")
    test_df = pd.read_csv(test_data_path, sep=';', encoding=ENCODING, index_col=0)
    test_df["idx"] = pd.Series(range(test_df.shape[0]))
    test_set = test_df[['text']]

    # Nettoyage des données
    print("Nettoyage...")
    test_set['text'] = cleaning_pdf_interest(test_set, 'text')

    # Pré-traitement
    print("Pré-traitement...")
    X = vectorizer.transform(test_set['text'])

    # Prédiction
    print("Prédictions...")
    test_df['is_interest'] = model.predict(X)

    # Sauvegarde des résultats
    print("Sauvegarde des résultats...")
    test_df.to_csv('results/predictions.csv')

    # Exemples
    print("Plots des exemples de classification...")

    # Tableau scanné prédit comme intéressant
    plot_items_scan_interesting(test_df, output_samples_dir)

    # Tableau non scanné prédit comme intéressant
    plot_items_raw_interesting(test_df, output_samples_dir)

    # Tableau scanné prédit comme non intéressant
    plot_items_scan_not_interesting(test_df, output_samples_dir)

    # Tableau non scanné prédit comme non intéressant
    plot_items_raw_not_interesting(test_df, output_samples_dir)

    print(f"Résultats enregistrés dans : {output_dir}")


@click.command()
@click.option("-a", "--action",
              required=True,
              type=click.Choice(['train', 'test']),
              help="Action à effectuer. Entraînement (train) ou test (test)")
@click.option("-d", "--data-file",
              required=True,
              type=click.Path(exists=True, file_okay=True, path_type=Path),
              help="Chemin vers le fichier csv des données")
@click.option("-o", "--output-dir",
              type=click.Path(dir_okay=True, path_type=Path),
              default="results",
              help="Dossier des résultats")
@click.option("-m", "--model",
              type=click.Path(exists=True, file_okay=True, path_type=Path),
              help="Chemin vers le fichier du modèle (action=test)")
@click.option("-v", "--vectorizer",
              type=click.Path(exists=True, file_okay=True, path_type=Path),
              help="Chemin vers le fichier du vectoriseur (action=test)")
def main(action: str, data_file: Path, output_dir: Path, model: Path, vectorizer: Path):
    makedirs("results")

    if action == "train":
        train(data_file, output_dir)
    else:
        if model is None:
            print("Error: Missing option '-m' / '--model'.")
        elif vectorizer is None:
            print("Error: Missing option '-v' / '--vectorizer'.")
        test(data_file, model, vectorizer, output_dir)


if __name__ == "__main__":
    main()
