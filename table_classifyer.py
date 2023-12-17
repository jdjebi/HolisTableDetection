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
from ml.plot import plot_dist_interest_table, plot_roc_curve
from ml.preprocess_func import cleaning_pdf_interest
from utils.utils import makedirs

TEST_SIZE = 0.3
RAMDOM_SATE = 42

path = "data/temp/train_dataset/csv/table_dataset.csv"


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

    # Sauvegarde des modèles
    print(f"\nSauvegarde des modèles...")
    joblib.dump(lr, 'results/table_classifier_models/lr_model.pk')
    joblib.dump(rf, 'results/table_classifier_models/rf_model.pkl')
    joblib.dump(svm, 'results/table_classifier_models/svm_model.pkl')


def main(train_data_path: Path):
    makedirs("results")

    train(train_data_path)


if __name__ == "__main__":
    main(path)
