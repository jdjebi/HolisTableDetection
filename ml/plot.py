from fitz import fitz
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from utils.typing_toolkit import StrPath


def plot_dist_interest_table(train_set):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=train_set, x='interest_table', hue='interest_table')
    plt.xlabel('Interest Table')
    plt.ylabel('Nombres')
    plt.title("Distribution des tableaux d'intérêts")
    # TODO: Faire en sorte que le chemin soit dynamique
    plt.savefig("results/plot_dist_interest_table")


def plot_roc_curve(y_test, y_prob, save_path):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 8))

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')

    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_pdf_page(path: StrPath, page: int, save_path: StrPath):
    with fitz.open(path) as pdf_document:
        page = pdf_document[int(page)]
        pix = page.get_pixmap(dpi=200)
        pix.save(save_path)
