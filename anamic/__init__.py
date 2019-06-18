from pathlib import Path
import logging

__version__ = '0.2.2'

from . import simulator
from . import imaging
from . import utils
from . import geometry
from . import fitter
from . import viz


def run_all_tests():

  try:
    import pytest
  except ImportError:
    logging.error("You need to install pytest to run tests.")
    return

  maskflow_dir = Path(__file__).parent

  if maskflow_dir.is_dir():
    pytest.main(["-v", str(maskflow_dir)])
  else:
    mess = f"anamic directory can't be found: {maskflow_dir}"
    logging.error(mess)
