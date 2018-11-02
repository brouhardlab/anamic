from tqdm import tqdm_notebook as tqdm
from joblib import Parallel, delayed

all_bar_funcs = {
    'tqdm': lambda args: lambda x: tqdm(x, **args),
    'txt': lambda args: lambda x: text_progessbar(x, **args),
    'False': lambda args: iter,
    'None': lambda args: iter,
}

def parallel_executor(use_bar='tqdm', **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        all_bar_funcs = {
            'tqdm': lambda args: lambda x: tqdm(x, **args),
            'False': lambda args: iter,
            'None': lambda args: iter,
        }
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type"%bar)
            return Parallel(**joblib_args)(bar_func(op_iter))
        return tmp
    return aprun
