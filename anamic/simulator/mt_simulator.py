import json
import datetime

import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.spatial.transform import Rotation
import tifffile
import matplotlib.pyplot as plt

from . import structure
from .. import viz
from .. import imaging
from .. import geometry


def dimers_builder(n_pf, mt_length_nm, taper_length_nm):
  long_dimer_distance = 8  # nm

  # Compute the number of rows
  n_rows = int(np.round(mt_length_nm / long_dimer_distance))

  # Create the dimers array
  dimers = np.ones((n_pf, n_rows))

  # Generate a tapered tip
  dimers = structure.generate_uniform_taper(dimers, taper_length_nm=taper_length_nm)
  return dimers


class MicrotubuleSimulator():
  """

  Args:
    dimers: Numpy array of dimension 2.
  """

  long_dimer_distance = 8  # nm

  enable_cached_positions = True
  cached_positions = {}

  def __init__(self, dimers):
    self.dimers = dimers

    self.positions = None
    self.discrete_image = None
    self.psf = None
    self.image = None

    self.parameters = {}

    self.parameters['date'] = datetime.datetime.now().isoformat()
    self.parameters['n_pf'] = self.dimers.shape[0]
    self.parameters['row'] = self.dimers.shape[1]

    # Set default parameters

    self.parameters['3d_z_rotation_angle'] = np.nan  # degrees
    self.parameters['projected_rotation_angle'] = np.nan  # degrees
    self.parameters['labeling_ratio'] = 0.1  # from 0 to 1

    self.parameters['pixel_size'] = 110  # nm/pixel
    self.parameters['x_offset'] = 1000  # nm
    self.parameters['y_offset'] = 1000  # nm

    self.parameters['psf_size'] = 135  # nm
    self.parameters['sigma_pixel'] = self.parameters['psf_size'] / self.parameters['pixel_size']

    self.parameters['tip_start'] = None
    self.parameters['tip_end'] = None

    self.parameters['signal_mean'] = 100
    self.parameters['signal_std'] = 40
    self.parameters['bg_mean'] = 50
    self.parameters['bg_std'] = 30
    self.parameters['noise_factor'] = 1

    self.parameters['snr_line_width'] = 3  # pixel
    self.parameters['snr'] = np.nan

  def build_all(self, apply_random_z_rotation=True, show_progress=True):

    # Build the geometry
    self.build_positions(apply_random_z_rotation=apply_random_z_rotation, show_progress=show_progress)
    self.label()
    self.project()
    self.random_rotation_projected()

    # Generate the image
    self.discretize_position()
    self.convolve()

    self.calculate_snr()

  # Methods to build the microtubule geometry.

  def build_positions(self, apply_random_z_rotation=True, show_progress=True):
    """Calculate the x, y and z positions of each dimers.

    Args:
      apply_random_z_rotation: boolean.
          Apply a random rotation to 3D positions along the Z axis (microtubule length).
      show_progress: boolean.
          Show a progress bar.
    """

    if MicrotubuleSimulator.enable_cached_positions:
      n_pf = self.dimers.shape[0]
      if self.dimers.shape[0] in MicrotubuleSimulator.cached_positions.keys():
        if self.dimers.shape[1] <= MicrotubuleSimulator.cached_positions[self.dimers.shape[0]]['row'].max():
          positions_to_keep = MicrotubuleSimulator.cached_positions[self.dimers.shape[0]]['row'] < self.dimers.shape[1]
          self.positions = MicrotubuleSimulator.cached_positions[self.dimers.shape[0]][positions_to_keep].copy()

    if self.positions is None:
      self.positions = structure.get_dimer_positions(self.dimers, show_progress=show_progress)
      n_pf = self.positions['pf'].max() + 1
      MicrotubuleSimulator.cached_positions[n_pf] = self.positions.copy()

    if apply_random_z_rotation:
      self._random_3d_z_rotation()

  def _random_3d_z_rotation(self):
    """Apply a random rotation parallell to the surface (along the z axis).

    This is to avoid having the seam always at the same location.
    """
    self.parameters['3d_z_rotation_angle'] = np.random.randn() * 360
    # pylint: disable=assignment-from-no-return
    rotation_angle = np.deg2rad(self.parameters['3d_z_rotation_angle'])
    rot_z = Rotation.from_euler('z', rotation_angle, degrees=False)
    self.positions[['x', 'y', 'z']] = rot_z.apply(self.positions[['x', 'y', 'z']].values)

  def label(self):
    """Apply a certain labeling ratio. This will add a column 'labeled' to `self.positions`.
    """

    self.positions['labeled'] = np.random.random(self.positions.shape[0]) < self.parameters['labeling_ratio']

  def project(self):
    """Project the 3D positions (x, y, z) on a 2D plan. The projection is done
    along the microtubule z axis so the microtubule is parallel to the "fake" coverslip.
    Projected coordinates are called `x_proj` and `y_proj`.
    """
    self.positions[['x_proj', 'y_proj']] = self.positions[['x', 'z']].copy()

  def random_rotation_projected(self):
    """Apply a random 2D rotation on `x_proj` and `y_proj`. New coordinates are called
    'x_proj_rotated' and 'y_proj_rotated'.
    """
    self.parameters['projected_rotation_angle'] = np.random.randn() * 360
    # pylint: disable=assignment-from-no-return
    random_angle = np.deg2rad(self.parameters['projected_rotation_angle'])
    rot_z = Rotation.from_euler('z', random_angle, degrees=False)
    rotation = rot_z.as_dcm()[:2, :2].T

    self.positions['x_proj_rotated'] = np.nan
    self.positions['y_proj_rotated'] = np.nan
    self.positions[['x_proj_rotated', 'y_proj_rotated']] = np.dot(self.positions[['x_proj', 'y_proj']], rotation)

  # Methods to generate the image

  def discretize_position(self):
    """Discretize dimer positions on an image. Image is stored in `self.discrete_image`.
    `self.positions` will contains two new columns: `x_pixel` and `y_pixel`.
    """

    # Discretize dimers positions onto an image
    x_max = int(np.round(self.positions['x_proj_rotated'].max() + 1))
    x_min = int(np.round(self.positions['x_proj_rotated'].min() - 1))
    y_max = int(np.round(self.positions['y_proj_rotated'].max() + 1))
    y_min = int(np.round(self.positions['y_proj_rotated'].min() - 1))

    x_bins = np.arange(x_min - self.parameters['x_offset'], x_max + self.parameters['x_offset'], self.parameters['pixel_size'])
    y_bins = np.arange(y_min - self.parameters['y_offset'], y_max + self.parameters['y_offset'], self.parameters['pixel_size'])

    # Select visible and labeled dimers
    selected_dimers = self.positions[self.positions['visible'] & self.positions['labeled']]

    # Bin dimers positions to a 2D grid (defined by pixel_size)
    self.discrete_image, _, _ = np.histogram2d(selected_dimers['x_proj_rotated'],
                                               selected_dimers['y_proj_rotated'],
                                               bins=[x_bins, y_bins])

    # Keep the width > height consistant
    if self.discrete_image.shape[1] < self.discrete_image.shape[0]:
      self.discrete_image = self.discrete_image.T
      self.positions[['x_proj', 'y_proj']] = self.positions.loc[:, ['y_proj', 'x_proj']]
      self.positions[['x_proj_rotated', 'y_proj_rotated']] = self.positions.loc[:, ['y_proj_rotated', 'x_proj_rotated']]
      x_bins, y_bins = y_bins, x_bins

    # We also save the dimers positions on the fine grid (unit is pixel)
    # The pixel_shift is not ideal but seems to be necessary...
    pixel_shift = -1
    self.positions.loc[:, 'x_pixel'] = np.digitize(self.positions['x_proj_rotated'], x_bins) + pixel_shift
    self.positions.loc[:, 'y_pixel'] = np.digitize(self.positions['y_proj_rotated'], y_bins) + pixel_shift

    # Save tips positions
    x1, x2, y1, y2 = structure.get_mt_tips(self.positions, coordinates_features=['y_pixel', 'x_pixel'])
    self.parameters['tip_start'] = (y1, x1)
    self.parameters['tip_end'] = (y2, x2)

  def generate_psf(self):
    """Generate a PSF from a Gaussian.
    """
    self.parameters['sigma_pixel'] = self.parameters['psf_size'] / self.parameters['pixel_size']
    kernel_size_pixel = int(self.parameters['sigma_pixel'] * 10)
    #pylint: disable=no-member
    gaussian_kernel_1d = signal.gaussian(kernel_size_pixel, std=self.parameters['sigma_pixel'])
    gaussian_kernel_1d = gaussian_kernel_1d.reshape(kernel_size_pixel, 1)
    self.psf = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)

  def convolve(self):
    """Convolve `self.discrete_image` with a PSF and add some Poisson
    noise.
    """

    # Get the PSF
    self.generate_psf()

    # Convolve the image with the PSF
    self.image = ndimage.convolve(self.discrete_image, self.psf, mode="constant")

    # Scale image to 0 - 1
    self.image /= self.image.max()

    # Bring signal to signal_mean and background to bg_mean
    self.image = self.image * (self.parameters['signal_mean'] - self.parameters['bg_mean']) + self.parameters['bg_mean']

    # Create shot noise
    # TODO: is it worh it?
    # image = np.random.poisson(image).astype('float')

    # Create read noise
    read_noise = np.random.normal(loc=0, scale=self.parameters['bg_std'], size=self.image.shape) * self.parameters['noise_factor']
    self.image += read_noise

  def calculate_snr(self):
    """Calculate the SNR of the microtubule signal on the generated image.

    The SNR is very sensitive to the line_width parameter.

    Args:
      line_width: float, used to select pixels belonging to the signal (pixel).
    """

    x1, x2, y1, y2 = structure.get_mt_tips(self.positions, coordinates_features=['y_pixel', 'x_pixel'])
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])

    corners = geometry.get_rectangle_from_middle_line(p1, p2, rectangle_width=self.parameters['snr_line_width'])
    mask = geometry.get_mask_from_polygon(self.image, corners)

    image_signal = self.image[mask]
    image_background = self.image[~mask]

    self.parameters['snr'] = imaging.compute_snr(image_signal.mean(),
                                                 image_signal.std(),
                                                 image_background.mean(),
                                                 image_background.std())
    return self.parameters['snr']

  # Methods to visualize positions or images.

  def visualize_2d_positions(self, x_feature, y_feature, show_all=True, show_labeled=True,
                             color_feature='pf', marker_size=30, x_offset=400):
    """Visualize 2D dimer positions.

    Args:
      x_feature: string, the x feature to be used.
      y_feature: string, the y feature to be used.
      show_all: boolean, show all visible dimers.
      show_labeled: boolean, show only labeled dimers.
      color_feature: string, feature to use to color dimers.
      marker_size: float, size of the dimer marker.
      x_offset: float, offset apply on the X axis to show only the tips.
    """

    axs = None
    if show_all and show_labeled:
      fig, axs = plt.subplots(nrows=2, figsize=(18, 6), constrained_layout=True)
    else:
      fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)

    if show_all:
      ax = axs[0] if show_labeled else ax
      # Visualize all dimers
      selected_dimers = self.positions[self.positions['visible']]
      viz.viz_2d_dimers_positions(ax, selected_dimers,
                                  x_feature=x_feature, y_feature=y_feature,
                                  color_feature=color_feature, marker_size=marker_size,
                                  x_offset=x_offset)

    if show_labeled:
      ax = axs[1] if show_all else ax
      # Visualize only labeled dimers
      selected_dimers = self.positions[self.positions['visible'] & self.positions['labeled']]
      viz.viz_2d_dimers_positions(axs[1], selected_dimers,
                                  x_feature=x_feature, y_feature=y_feature,
                                  color_feature=color_feature, marker_size=marker_size,
                                  x_offset=x_offset)
    return fig

  def show_positions(self, color_feature_name='pf', size=0.4):
    # Show 3D position
    return viz.viz_dimer_positions(self.positions, size=size,
                                   color_feature_name=color_feature_name)

  def show_psf(self):
    _, ax = plt.subplots(figsize=(5, 5))
    viz.imshow_colorbar(self.psf, ax)

  def show_image(self, tip_marker_size=80):
    _, ax = plt.subplots(figsize=(8, 8))
    viz.imshow_colorbar(self.image, ax)
    viz.show_tips(ax, self.positions, coordinates_features=['y_pixel', 'x_pixel'], marker_size=tip_marker_size)

  # Utility methods

  def save_positions(self, fpath):
    """Save `self.positions` to a CSV file.
    """
    self.positions.to_csv(fpath)

  def save_parameters(self, fpath):
    """Save `self.parameters` to a JSON file.
    """
    with open(fpath, 'w') as fp:
      json.dump(self.parameters, fp, indent=2, sort_keys=True)

  def save_image(self, fpath):
    """Save `self.image` to a TIFF file.
    """
    tifffile.imsave(fpath, self.image)
