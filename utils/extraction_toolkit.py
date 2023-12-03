""" Module de fonctions d'extraction des données dans un fichier pdf. """

from pathlib import Path
from typing import Union, List, Tuple, Dict

from tqdm import tqdm
from fitz import fitz
from fitz.fitz import Page, Rect

from preprocess.text_preprocess import clean_and_flatten_text
from utils.utils import generate_unique_id


def get_page_annotations(page: Page) -> List:
    return list(page.annots())


def check_if_annotation_has_image(annot_rect: Rect, images_list: List, page: Page) -> Tuple:
    annot_im = None
    annot_im_rect = None
    annot_has_image = False

    for im in images_list:
        im_rect = page.get_image_rects(im)[0]
        if im_rect.intersects(fitz.Rect(annot_rect)):
            annot_im = im
            annot_has_image = True
            annot_im_rect = im_rect
            break

    return annot_has_image, annot_im, annot_im_rect


def extract_document_annotate_text(pdf_path: Union[Path, str], verbose: int = 1) -> List[Dict]:
    dict_document = []

    with fitz.open(pdf_path) as pdf_document:

        if verbose == 1:
            print(f">>> Doc {pdf_path}")

        for i, page in enumerate(pdf_document):

            annotations = get_page_annotations(page)

            # Vérifier s'il y a une annotation
            n_annotations = len(annotations)

            if n_annotations > 0:

                interest_item = 1

                # Récupération des images de la page

                images_list = page.get_images()
                has_images = True if len(images_list) > 0 else False

                # Parcours des annotations

                for annot in annotations:

                    annot_rect = annot.rect

                    # Vérifier si l'annotation encadre un tableau scanné ou non
                    annot_is_scan = False

                    if has_images:
                        # On vérifie il y a une image à l'intérieur de la rect de l'annotation
                        annot_has_image, im, im_rect = check_if_annotation_has_image(annot_rect, images_list, page)

                        if annot_has_image:
                            annot_is_scan = True

                if verbose == 1:
                    if annot_is_scan:
                        print(f"\033[1;32m\t- Page {i} contient une image annotée")
                        print(f"\033[0m", end="")
                    else:
                        print(f"\033[1;36m\t- Page {i} contient tableau de données brute.")
                        print(f"\033[0m", end="")

                # Extraction du texte dans la region de l'annotation

                region_text = page.get_text("text", clip=annot_rect)
                text = clean_and_flatten_text(region_text)

                # Vérification
                if verbose == 1:
                    if annot_is_scan and len(text) > 0:
                        print(
                            f"\033[1;33m\t- !Page {i} incohérence : Déterminé comme image mais contient {len(text)} "
                            f"caractère(s). Il s'agit donc de données brutes normalement.")

            else:
                interest_item = 0

                if verbose == 1:
                    print(f"\033[1;31m\t- Page {i} ne contient pas d'annotation")
                    print(f"\033[0m", end="")

                text = page.get_text()
                text = clean_and_flatten_text(text)

                annot_is_scan = False

            # Sauvegarde des données

            dict_document.append({
                "doc_id": generate_unique_id(8),
                "path": str(pdf_path),
                "page": i,
                "is_scan": annot_is_scan,
                "text": text,
                "interest_table": interest_item
            })

    return dict_document


def extract_document_text(pdf_path: Union[Path, str], verbose: int = 1) -> List[Dict]:
    dict_document = []

    with fitz.open(pdf_path) as pdf_document:
        for i, page in enumerate(pdf_document):
            # Récupération des images de la page
            images_list = page.get_images()
            is_scan = True if len(images_list) > 0 else False

            # Récupération du texte de la page
            text = page.get_text()
            text = clean_and_flatten_text(text)

            dict_document.append({
                "doc_id": generate_unique_id(8),
                "path": str(pdf_path),
                "page": i,
                "is_scan": is_scan,
                "text": text,
            })

    return dict_document


def extract_pdf_data(pdf_files: List[Path], versbose: int = 0, annotate: bool = True) -> List[Dict]:
    data = []

    if versbose == 0:
        with tqdm(total=len(pdf_files), unit="pdf", desc=f"Extraction") as progress_bar:
            for pdf_file_path in pdf_files:

                if annotate:
                    dict_document: List = extract_document_annotate_text(pdf_file_path, verbose=versbose)
                else:
                    dict_document: List = extract_document_text(pdf_file_path, verbose=versbose)

                data += dict_document
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "pages": len(dict_document),
                    "pdf": pdf_file_path.stem
                })
    else:
        for pdf_file_path in pdf_files:
            if annotate:
                dict_document: List = extract_document_annotate_text(pdf_file_path, verbose=versbose)
            else:
                dict_document: List = extract_document_text(pdf_file_path, verbose=versbose)
            data += dict_document

    return data
