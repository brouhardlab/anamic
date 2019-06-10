import numpy as np


def compute_snr(signal_mean, signal_std, bg_mean, bg_std):
  """Compute the SNR of a given object in an image.

  Args:
      signal_mean: float, mean signal value.
      signal_std: float, std signal value.
      bg_mean: float, mean background value.
      bg_std: float, std background value.
  """
  return (signal_mean - bg_mean) / np.sqrt(signal_std**2 + bg_std**2)
