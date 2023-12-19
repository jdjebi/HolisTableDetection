from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from ml.constants import RAMDOM_SATE, PLOT_DIR
from ml.log import show_coef
from ml.plot import plot_roc_curve


def train_lr_model(X_train: Any, y_train: Any,
                   X_test: Any, y_test: Any,
                   features_names: Any, output_dir: Path) -> LogisticRegression:
    lr = LogisticRegression(random_state=RAMDOM_SATE)
    lr = lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(classification_report(y_test, y_pred))
    y_prob = lr.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_prob, output_dir / PLOT_DIR / "lr_plot_curve")
    show_coef(lr.coef_[0], features_names)
    return lr


def train_rf_model(X_train: Any, y_train: Any,
                   X_test: Any, y_test: Any,
                   features_names: Any, output_dir: Path) -> RandomForestClassifier:
    rf = RandomForestClassifier(min_samples_split=5, max_depth=10, random_state=RAMDOM_SATE)
    rf = rf.fit(X_train, y_train)
    y_pred2 = rf.predict(X_test)
    print(classification_report(y_test, y_pred2))
    y_prob = rf.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_prob, output_dir / PLOT_DIR /"rf_plot_curve")
    show_coef(rf.feature_importances_, features_names)
    return rf


def train_svm_model(X_train: Any, y_train: Any,
                    X_test: Any, y_test: Any,
                    features_names: Any, output_dir: Path) -> RandomForestClassifier:
    svm = LinearSVC(random_state=42)
    svm = svm.fit(X_train, y_train)
    y_pred3 = svm.predict(X_test)
    print(classification_report(y_test, y_pred3))
    y_scores = svm.decision_function(X_test)
    plot_roc_curve(y_test, y_scores, output_dir / PLOT_DIR / "svm_plot_curve")
    show_coef(svm.coef_[0], features_names)
    return svm
