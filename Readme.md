# Les Geeks ! 

## DETECTION ET EXTRACTION DE TABLES DE DOCUMENT NON STRUCTURES

Ce readme explique la procédure à suivre pour utiliser pour détecter et extraite les table de documents PDF.

## Installation

Pour bien démarrer, créer un environment virtuelle pour installer les dépendances.

```
python -m venv venv
```

Une fois l'environnement créé, procédez à l'installation des dépendances.

```
python install -r requirements.txt
```

Ensuite l'installer l'outil Tesserac OCR de Google qui permettra d'extraire le texte des tables scannées.

Tesseract: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

À ce stade, le projet est prêt pour extraire les données, entraîner et tester un modèle de détection de table.

## Entraînement

### Extraction des données

#### Vue globale

L'extraction des données ce fait par l'exécution successive de plusieurs scripts. Chaque script représente une
étape de la pipeline de données. L'intérêt de cette approche est de ne pas avoir à re-exécuter toute la pipeline
en cas de problème. Cette approche est valable uniquement pour l'entraînement. Lors de l'inférence la pipeline est 
exécutée en une seule fois ([extract_for_inference.py](extractation_for_inference.py)).

#### Pipeline 

##### 1. Extraction du texte des tables brutes et annotations automatiques des données

Cette étape est réalisée par le script [pdf_text_data_extraction.py](extract_pdf_text.py). Elle parcourt les pdf correspond
au dossier spécifié dans la variable SRC_PATH du script et extrait le texte brute des tableaux annotés.
L'ensemble de données est ajouté dans un fichier [datasets/csv/pdf_raw_text.csv](datasets%2Fcsv%2Fpdf_raw_text.csv)

##### 2. Extraction des images des tableaux scannés

Réalisée par [pdf_scan_table_pages_extraction.py](extract_pdf_scan.py), cette étape permet d'extraire sous
forme d'images les pages contenant des tableaux scannés. Il récupère les pages concernées dans le fichier [datasets/csv/pdf_raw_text.csv](datasets%2Fcsv%2Fpdf_raw_text.csv),
puis sélection les pages ayant été marquées comme scanne. 

Après la sélection il enregistre les images dans un dossier 
[datasets/scan_table_page_images](datasets%2Fscan_table_page_images). La correspondance entre pdf (et leur page) est
enregistrée dans le fichier [datasets/csv/scan_table_page_images.csv](datasets%2Fcsv%2Fscan_table_page_images.csv)

A ce stade, on peut appliquer notre de détection des tables sur le dossier [datasets/scan_table_page_images](datasets%2Fscan_table_page_images)
pour détecter les tables et leur appliquer un OCR (L'OCR est fait en utilisant tesseract de google)

##### 3. Détection des tables

Lors de cette étape, l'exécution du script [table_detector.py](table_detector.py)
permet d'appliquer le modèle pré-entrainé Table-Transformer de Microsoft pour détecter les tables. Ensuite Tesseract OCR
est appliquée sur les tableaux détectés 
