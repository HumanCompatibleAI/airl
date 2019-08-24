"""Utilities for distributed experiments."""
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from functools import wraps
import itertools
import os
import sys
import traceback
from typing import Callable, Optional


def path_from_ordered_dict(d: OrderedDict) -> str:
  """Generates a hierarchal path by joining key-values sequentially.

  For example, `OrderedDict({1: 10, 2: 20})` becomes "1/10/2/20".
  (This mimics the hierarchical directory structure generated by GNU
  parallel's `--header --results` options.)
  """
  return os.path.join(*itertools.chain.from_iterable(d.items()))


@contextmanager
def redirect_stdout(path):
  temp = sys.stdout
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      sys.stdout = f
      yield
  finally:
    sys.stdout = temp


@contextmanager
def redirect_stderr(path):
  temp = sys.stderr
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      sys.stderr = f
      yield
  finally:
    sys.stderr = temp


def redirect_to_files(
  f: Callable,
  stdout_path: Optional[str] = None,
  stderr_path: Optional[str] = None,
) -> Callable:
  """A decorator wrapping `f` in contexts redirecting console output to files.

  Exception tracebacks are always printed to the original stderr.
  """
  @wraps(f)
  def inner(*args, **kwargs):
    with ExitStack() as stack:
      if stdout_path is not None:
        stack.enter_context(redirect_stdout(stdout_path))
      if stderr_path is not None:
        stack.enter_context(redirect_stderr(stderr_path))
      try:
        return f(*args, **kwargs)
      except:  # noqa: E901
        traceback.print_exc()
        raise

  return inner


def args_single_to_star(f: Callable) -> Callable:
  @wraps(f)
  def inner(args):
    return f(*args)
  return inner
