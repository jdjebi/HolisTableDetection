from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from config.constants import ENCODING
from ml.log import show_coef
from ml.plot import plot_dist_interest_table, plot_roc_curve, plot_pdf_page
from ml.preprocess_func import cleaning_pdf_interest
from utils.utils import makedirs

TEST_SIZE = 0.3
RAMDOM_SATE = 42

train_path = "data/temp/train_dataset/csv/table_dataset.csv"


# TODO: Dans l'extraction, trouver un moyen pour préciser qu'une page contient à la fois du texte et des images (
#  text_only: False)
def train(train_data_path: Path):
    makedirs('results/table_classifier_models', exist_ok=True)

    print("# Entraînement")
    print(f"Données : {train_data_path}\n")

    # Chargement des données
    train_df = pd.read_csv(train_data_path, delimiter=';', encoding=ENCODING)
    train_df['interest_table'] = train_df['interest_table'].astype(int)
    train_set = train_df[['text', 'interest_table']]

    # Distribution de données par tableau d'intérêt
    plot_dist_interest_table(train_set)

    # Nettoyage des données
    print(f"Nettoyage des données...")
    train_set_cleaned = train_set.copy()
    train_set_cleaned['text'] = cleaning_pdf_interest(train_set_cleaned, 'text')

    # Observation
    n_words = len([word for line in train_set_cleaned["text"].tolist() for word in line.split()])
    print(f"Nombre de mots: {n_words}")

    # Division des données
    X = train_set_cleaned['text'].tolist()
    y = train_set_cleaned['interest_table'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RAMDOM_SATE, stratify=y)

    # Vectorisation
    tfidf_vector = TfidfVectorizer(stop_words='english', max_features=None, max_df=0.5, min_df=2, ngram_range=(1, 1))
    X_train = tfidf_vector.fit_transform(X_train)
    X_test = tfidf_vector.transform(X_test)

    features_names = tfidf_vector.get_feature_names_out()
    print(f"Quelques features : {features_names[:5]}")

    # Entraînement
    print(f"\nEntraînement des modèles...")

    print("## LogisticRegression")
    lr = LogisticRegression(random_state=RAMDOM_SATE)
    lr = lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(classification_report(y_test, y_pred))
    y_prob = lr.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_prob, "results/lr_plot_curve")
    show_coef(lr.coef_[0], features_names)

    print("## RandomForestClassifier")
    rf = RandomForestClassifier(min_samples_split=5, max_depth=10, random_state=RAMDOM_SATE)
    rf = rf.fit(X_train, y_train)
    y_pred2 = rf.predict(X_test)
    print(classification_report(y_test, y_pred2))
    y_prob = rf.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_prob, "results/rf_plot_curve")
    show_coef(rf.feature_importances_, features_names)

    print("## LinearSVC")
    svm = LinearSVC(random_state=42)
    svm = svm.fit(X_train, y_train)
    y_pred3 = svm.predict(X_test)
    print(classification_report(y_test, y_pred3))
    y_scores = svm.decision_function(X_test)
    plot_roc_curve(y_test, y_scores, "results/svm_plot_curve")
    show_coef(svm.coef_[0], features_names)

    # Sauvegarde
    print(f"\nSauvegarde du vectoriseur...")
    joblib.dump(tfidf_vector, 'results/table_classifier_models/tfidf_vector.pk')

    print(f"Sauvegarde des modèles...")
    joblib.dump(lr, 'results/table_classifier_models/lr_model.pkl')
    joblib.dump(rf, 'results/table_classifier_models/rf_model.pkl')
    joblib.dump(svm, 'results/table_classifier_models/svm_model.pkl')

    print("Sauvegarde terminée! Dossier de sauvegarde results/table_classifier_models")


def test(test_data_path: Path):
    print("# Test")
    print(f"Données : {test_data_path}\n")

    # Chargement du modèle et du vectoriseur
    print("Chargement du modèle...")
    model = joblib.load('results/table_classifier_models/lr_model.pkl')
    vectorizer = joblib.load('results/table_classifier_models/tfidf_vector.pk')

    # Chargement des données
    print("Chargement des données...")
    test_df = pd.read_csv(test_data_path, sep=';', encoding=ENCODING, index_col=0)
    test_df["idx"] = pd.Series(range(test_df.shape[0]))
    test_set = test_df[['text']]

    # Nettoyage des données
    test_set['text'] = cleaning_pdf_interest(test_set, 'text')

    # Pré-traitement
    print("Pré-traitement...")
    X = test_set['text'].tolist()
    X = vectorizer.transform(X)

    # Prédiction
    print("Prédictions...")
    y_pred = model.predict(X)
    test_df['is_interest'] = y_pred

    # Sauvegarde des résultats
    print("Sauvegarde des résultats...")
    test_df.to_csv('results/predictions.csv')

    # Exemples
    print("Plots des exemples de classification...")

    makedirs(Path("results/samples"), remove_ok=True)

    # Tableau scanné prédit comme intéressant
    items_scan_interesting = test_df[(test_df.is_scan == True) & (test_df.is_interest == 1)]
    item = items_scan_interesting.iloc[0, :]
    suffix = f"{item.idx}_p_{item.page}"
    plot_pdf_page(item.path, item.page, f"results/samples/items_scan_interesting_{suffix}.jpg")

    # Tableau non scanné prédit comme intéressant
    items_scan_interesting = test_df[(test_df.is_scan == False) & (test_df.is_interest == 1)]
    item = items_scan_interesting.iloc[0, :]
    suffix = f"{item.idx}_p_{item.page}"
    plot_pdf_page(item.path, item.page, f"results/samples/items_raw_interesting_{suffix}.jpg")

    # Tableau scanné prédit comme non intéressant
    items_scan_interesting = test_df[(test_df.is_scan == True) & (test_df.is_interest == 0)]
    item = items_scan_interesting.iloc[0, :]
    suffix = f"{item.idx}_p_{item.page}"
    plot_pdf_page(item.path, item.page, f"results/samples/items_scan_not_interesting_{suffix}.jpg")

    # Tableau non scanné prédit comme non intéressant
    items_scan_interesting = test_df[(test_df.is_scan == False) & (test_df.is_interest == 0)]
    item = items_scan_interesting.iloc[0, :]
    suffix = f"{item.idx}_p_{item.page}"
    plot_pdf_page(item.path, item.page, f"results/samples/items_raw_not_interesting_{suffix}.jpg")


def main(train_data_path: Path):
    makedirs("results")

    # train(train_data_path)

    test("data/temp/test_dataset/csv/table_dataset.csv")


if __name__ == "__main__":
    main(train_path)
