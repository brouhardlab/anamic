import numpy as np
import anamic


def test_dimers_builder():
  n_pf = 11
  mt_length_nm = 100  # nm
  taper_length_nm = 50  # nm
  dimers = anamic.simulator.dimers_builder(n_pf, mt_length_nm, taper_length_nm)
  assert dimers.shape == (11, 12)
  assert dimers.sum() < 132


def test_mt_simulator():
  n_pf = 11
  mt_length_nm = 100  # nm
  taper_length_nm = 0  # nm

  dimers = anamic.simulator.dimers_builder(n_pf, mt_length_nm, taper_length_nm)

  # Set parameters for the image generation.
  parameters = {}
  parameters['labeling_ratio'] = 0.1  # from 0 to 1

  parameters['pixel_size'] = 110  # nm/pixel
  parameters['x_offset'] = 1500  # nm
  parameters['y_offset'] = 1500  # nm

  parameters['psf_size'] = 135  # nm

  parameters['signal_mean'] = 700
  parameters['signal_std'] = 100
  parameters['bg_mean'] = 500
  parameters['bg_std'] = 24
  parameters['noise_factor'] = 1

  parameters['snr_line_width'] = 3  # pixel

  ms = anamic.simulator.MicrotubuleSimulator(dimers)
  ms.parameters.update(parameters)

  # Build the geometry.
  ms.build_positions(apply_random_z_rotation=True, show_progress=True)
  ms.label()
  ms.project()
  ms.random_rotation_projected()

  # Generate the image
  ms.discretize_position()
  ms.convolve()

  snr = ms.calculate_snr()

  cols = ['row', 'pf', 'x', 'y', 'z', 'visible', 'labeled', 'x_proj', 'y_proj', 'x_proj_rotated', 'y_proj_rotated', 'x_pixel', 'y_pixel']
  assert list(ms.positions.columns) == cols
  assert ms.positions.shape == (132, 13)
  assert isinstance(snr, np.float64)
  assert ms.image.shape == (27, 28)
