from datetime import timedelta
import uuid
import numpy as np
import os
import json
import random


def format_time(seconds):
    if seconds is None:
        return "-:--:--"
    return str(timedelta(seconds=int(seconds)))


def get_unique_id(existing:list, pre_fix:str, random_seed:int=123) -> str:
    """Returns a unique str not in the existing list

    Args:
        existing (list): A list
        pre_fix (str): A str the uniqe str starts with
    Returns:
        str: A str not in the list
    """
    random.seed(random_seed)
    while True:
        rbits = random.getrandbits(128)
        candidate = pre_fix + uuid.UUID(int=rbits, version=4).hex
        if candidate not in existing:
            return candidate


def all_numbers_and_finite(arr: np.ndarray) -> bool:
    """Check that all elements in a numpy array are finite numbers
    Args:
        arr (np.ndarray)

    Returns:
        bool: When the check is True
    """
    # 1) Can it be cast to float?
    try:
        f = arr.astype(float)
    except (ValueError, TypeError):
        return False
    # 2) Are all entries finite?
    return np.isfinite(f).all()

        

def save_dict_to_disk(data:dict, output_dir:str, file_name:str) -> None:
    """
    Save the data dict to file_name in the given directory 
    under the current working directory

    Args:
        data (dict)
        output_dir (str)
        file_name (str) 
    """
    output_dir = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, file_name)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

