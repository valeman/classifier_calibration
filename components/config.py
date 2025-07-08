SEED = 123456789


import logging
import time
class TimeDiffFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.last_time = time.time()

    def format(self, record):
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        record.elapsed = f"{elapsed:.3f}s"
        record.created_time = self.formatTime(record, self.datefmt)
        return super().format(record)

# Define format including current time and elapsed time
formatter = TimeDiffFormatter(fmt='[%(created_time)s | +%(elapsed)s] %(message)s',
                              datefmt='%H:%M:%S')

# Configure logger
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [handler]