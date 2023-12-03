import os
import string
import uuid
from pathlib import Path
import random
from typing import Union, List, Any

import pandas as pd


def get_pdf_files(dir_path: Union[Path, str]) -> List[Path]:
    # TODO : Utiliser path.glob
    return [Path(dir_path) / path for path in os.listdir(dir_path)]


def makedirs(dir_path: Union[Path, str], exist_ok: bool = True):
    os.makedirs(dir_path, exist_ok=exist_ok)


def save_wth_dataframe(data: Any, save_path: Union[Path, str], encoding: str = "utf-8", sep=";") -> pd.DataFrame:
    df = pd.DataFrame(data)
    df.to_csv(save_path, sep=sep, encoding=encoding)
    return df


def generate_unique_id(size: int = 5):
    unique_id = str(uuid.uuid4())[:size]
    return unique_id
