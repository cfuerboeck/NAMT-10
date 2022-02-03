# coding: utf-8
import hashlib
import json
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np


def bytes_to_gigabytes(x: int) -> float:
    return np.round(x / (1024 ** 3), 2)


def get_file_size(file_path: str) -> float:
    assert os.path.isfile(file_path)
    return bytes_to_gigabytes(os.path.getsize(file_path))


def get_disk_usage(path: str) -> Tuple[str, float, float, float]:
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    total, used, free = shutil.disk_usage(path)
    return (
        path,
        bytes_to_gigabytes(total),
        bytes_to_gigabytes(used),
        bytes_to_gigabytes(free),
    )


def hash_string(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def hash_dict(x: Dict) -> str:
    return hash_string(json.dumps(x, sort_keys=True))
