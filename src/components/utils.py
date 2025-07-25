from datetime import timedelta, datetime
import uuid
import numpy as np
import os
import json
import random
import signal
import time

class TimeoutException(Exception):
    """Custom exception used specifically for signaling timeouts."""
    pass

def handler(signum, frame):
    raise TimeoutException("Run timed out after time limit.")

def run_with_timeout(seconds, func, *args, **kwargs):
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)  # Disable the alarm
        signal.signal(signal.SIGALRM, original_handler)

def format_time(seconds):
    if seconds is None:
        return "-:--:--"
    return str(timedelta(seconds=int(seconds)))

def time_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

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
    create_pwd_dir(output_dir)
    filename = os.path.join(output_dir, file_name)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_dict(output_dir: str, file_name: str) -> dict:
    """
    Load a dictionary from a JSON dumped .txt file.

    Parameters
    ----------
    output_dir : str
        Subdirectory (under cwd) where the file lives.
    file_name : str
        Name of the .txt file (e.g. "mydict.txt").

    Returns
    -------
    Dict
        The dictionary that was saved.

    Raises
    ------
    FileNotFoundError
        If the target file does not exist.
    ValueError
        If the file's contents aren't a JSON object.
    json.JSONDecodeError
        If the file isn't valid JSON.
    """
    path = os.path.join(os.getcwd(), output_dir, file_name)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find file at {path!r}")

    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object (dict) in {path!r}, got {type(obj)}")
    return obj


def create_pwd_dir(path):
    OUTPUT_DIR = os.path.join(os.getcwd(), path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

def get_subdirs(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]