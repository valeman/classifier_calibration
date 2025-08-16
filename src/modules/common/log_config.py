import logging
import time
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def log_progress_snapshot(progress, label="Evaluate"):
    with progress._lock:
        task_infos = []
        for task in progress.tasks:
            name = task.description
            task_infos.append(f"{name}")
        line = "\n".join(task_infos)
        logging.getLogger().info(f"\n{'-'*25} {label} {'-'*25} \n{line}\n")


class TimeDiffFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.last_time = time.time()

    def format(self, record):
        now = time.time()
        delta = now - self.last_time
        self.last_time = now
        record.elapsed = f"{delta:.3f}s"
        record.asctime = self.formatTime(record, self.datefmt)
        return super().format(record)


def configure_logger() -> logging.Logger:
    logging.captureWarnings(True)
    console = Console()

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=False,
        show_level=False,
        markup=True,
    )

    fmt = "[%(asctime)s | +%(elapsed)s] %(message)s"
    formatter = TimeDiffFormatter(fmt=fmt, datefmt="%H:%M:%S")
    rich_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [rich_handler]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    return logging.getLogger("app"), progress
