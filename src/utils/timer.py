# src/utils/timer.py
"""
timer.py
--------

Simple timing utilities for benchmarking pipeline steps.

Responsibilities:
- Context manager for measuring execution time
- Utility for timing GPU/CPU operations

Tools:
- time

TODO:
- Add support for PyTorch CUDA events
"""

import time
from functools import wraps


def timeit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = fn(*args, **kwargs)
        print(f"{fn.__name__} took {time.time() - start:.2f}s")
        return out
    return wrapper
