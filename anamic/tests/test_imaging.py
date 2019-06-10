from pathlib import Path

import numpy.testing as npt
import anamic

root_dir = Path(__file__).parent
data_dir = root_dir / "data"


def test_compute_snr():
  signal_mean = 200
  signal_std = 15
  bg_mean = 80
  bg_std = 25
  snr = anamic.imaging.compute_snr(signal_mean, signal_std, bg_mean, bg_std)

  true_snr = 4.11596604342

  print(f"SNR = {snr}")
  print(f"True SNR = {true_snr}")

  npt.assert_almost_equal(true_snr, snr)


def test_get_pixel_size():
  image_path = data_dir / "microtubule_XY.tif"
  pixel_size = anamic.imaging.get_pixel_size(image_path)

  true_pixel_size = 0.107

  print(f"Pixel Size = {pixel_size}")
  print(f"True Pixel Size = {true_pixel_size}")

  npt.assert_almost_equal(true_pixel_size, pixel_size)
