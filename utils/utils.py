import os
import shutil
import uuid
from pathlib import Path
from typing import Union, List, Any

import pandas as pd

from config.constants import ENCODING


def get_pdf_files(dir_path: Union[Path, str]) -> List[Path]:
    # TODO : Utiliser path.glob
    return [Path(dir_path) / path for path in os.listdir(dir_path)]


def makedirs(dir_path: Union[Path, str], remove_ok: bool = False, exist_ok: bool = True):
    if remove_ok:
        remove_dir_if_exists(dir_path)
    os.makedirs(dir_path, exist_ok=exist_ok)


def save_wth_dataframe(data: Any, save_path: Union[Path, str], encoding: str = "utf-8", sep=";") -> pd.DataFrame:
    df = pd.DataFrame(data)
    df.to_csv(save_path, sep=sep, encoding=encoding)
    return df


def save(data: Any, save_path: Path):
    return save_wth_dataframe(data, save_path, encoding=ENCODING)


def generate_unique_id(size: int = 5):
    unique_id = str(uuid.uuid4())[:size]
    return unique_id


def remove_dir_if_exists(path: Path):
    if path.exists():
        shutil.rmtree(path)


def make_and_remove_dir_if_exists(path: Path):
    remove_dir_if_exists(path)
    makedirs(path)
