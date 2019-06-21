import numpy as np
import anamic


def long_fn(arg1, arg2=0):
  import time
  time.sleep(0.1)
  return arg1 + arg2


def test_parallelized_single():
  args = np.random.randint(0, 50, size=(10,))
  results = anamic.utils.parallelized(long_fn, args, args_type='single', progress_bar=True, n_jobs=2)
  assert np.sum(results) == np.sum(args)


def test_parallelized_list():
  args = np.random.randint(0, 50, size=(10, 2))
  results = anamic.utils.parallelized(long_fn, args, args_type='list', progress_bar=True, n_jobs=2)
  assert np.sum(results) == np.sum(args)


def test_parallelized_dict():
  args = [{'arg1': val1, 'arg2': val2} for val1, val2 in np.random.randint(0, 50, size=(10, 2))]
  raw_args = np.array([list(arg.values()) for arg in args])
  results = anamic.utils.parallelized(long_fn, args, args_type='dict', progress_bar=True, n_jobs=2)
  assert np.sum(results) == np.sum(raw_args)
