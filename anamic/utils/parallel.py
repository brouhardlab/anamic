from tqdm.auto import tqdm
from joblib import Parallel


def parallel_executor(use_bar='tqdm', **joblib_args):
  def aprun(bar_name=use_bar, **tq_args):
    all_bar_funcs = {
        # pylint: disable=undefined-variable
        'tqdm': lambda args: lambda x: tqdm(x, **args),
        'False': lambda args: iter,
        'None': lambda args: iter,
    }

    def tmp(op_iter):
      if str(bar_name) in all_bar_funcs.keys():
        bar_func = all_bar_funcs[str(bar_name)](tq_args)
      else:
        raise ValueError(f"Value {bar_name} not supported as bar type")
      return Parallel(**joblib_args)(bar_func(op_iter))
    return tmp
  return aprun
