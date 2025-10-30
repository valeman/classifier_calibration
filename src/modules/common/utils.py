from datetime import timedelta, datetime
import uuid
import numpy as np
import os
import json
import random
import signal


def get_max_ram_mib() -> int:
    """
    Parse /proc/meminfo on Linux and return max physical RAM in MiB.
    """
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                parts = line.split()
                total_kib = int(parts[1])
                total_bytes = total_kib * 1024
                return total_bytes // (1024**2)
    raise RuntimeError("MemTotal not found in /proc/meminfo")


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


def get_unique_id(existing: list, pre_fix: str, random_seed: int = 123) -> str:
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


def save_dict_to_disk(data: dict, output_dir: str, file_name: str) -> None:
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
    with open(filename, "w", encoding="utf-8") as f:
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

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object (dict) in {path!r}, got {type(obj)}")
    return obj


def create_pwd_dir(path):
    OUTPUT_DIR = os.path.join(os.getcwd(), path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def get_subdirs(path):
    return [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]

def perturb_probs(y_prob: np.ndarray, delta: float, random_state=None) -> np.ndarray:
    """
    Perturb each two-class probability row by a small random epsilon in [-delta, +delta],
    adding epsilon to probability of class 0 and subtracting it from class 1,
    while guaranteeing 0 <= probs <= 1 and that each row sums to 1.

    Parameters
    ----------
    y_prob : np.ndarray
        Array of shape (n, 2). Rows are probability vectors [p0, p1].
    delta : float
        Maximum absolute perturbation (epsilon lies in [-delta, +delta] but clipped
        per-row so bounds are respected).
    random_state : None | int | np.random.Generator
        Optional RNG seed or np.random.Generator for reproducibility.

    Returns
    -------
    np.ndarray
        New array of shape (n, 2) with perturbed probabilities.
    """
    y = np.asarray(y_prob, dtype=float)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("y_prob must be shape (n, 2)")

    # normalize rows to sum to 1 (safe-guard)
    row_sums = y.sum(axis=1)
    if np.any(row_sums == 0):
        raise ValueError("One or more rows sum to zero and cannot be normalized.")
    y = y / row_sums[:, None]

    p0 = y[:, 0]
    p1 = y[:, 1]  # equals 1 - p0, but keep for clarity

    # allowed epsilon per row to keep both probabilities inside [0,1] is:
    # eps >= -p0  (so p0+eps >= 0) and eps <= p1  (so p1-eps >= 0)
    low_allowed = np.maximum(-delta, -p0)
    high_allowed = np.minimum(delta, p1)

    # fix potential tiny numerical flips so high_allowed >= low_allowed
    high_allowed = np.maximum(high_allowed, low_allowed)

    # RNG
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    eps = rng.uniform(low_allowed, high_allowed)

    p0_new = p0 + eps
    # ensure exact sum 1 to avoid drift
    p1_new = 1.0 - p0_new

    # final safety clipping (shouldn't be needed, but harmless)
    p0_new = np.clip(p0_new, 0.0, 1.0)
    p1_new = np.clip(p1_new, 0.0, 1.0)

    return np.vstack((p0_new, p1_new)).T
