"""Controlled cyclic GC at stream boundaries, with restoration on exceptions."""
from contextlib import contextmanager
from functools import wraps
import gc


@contextmanager
def stream_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def ordered_run(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with stream_gc():
            return function(*args, **kwargs)
    return wrapped
