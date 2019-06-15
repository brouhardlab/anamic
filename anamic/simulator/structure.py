from pathlib import Path

import numpy as np
import tqdm
import pandas as pd
from scipy.spatial.transform import Rotation

DATA_DIR = Path(__file__).parents[1] / "data"


def get_structure_parameters():
  struct = pd.read_csv(DATA_DIR / 'mt_structure.csv')
  struct = struct.set_index('n_pf', drop=True)
  return struct


def generate_uniform_taper(dimers, taper_length_nm):

  if taper_length_nm == 0:
    return dimers

  long_dimer_distance = 8  # nm
  taper_length = int(np.round(taper_length_nm) / long_dimer_distance)
  n_pf = dimers.shape[0]

  if taper_length < 2:
    return dimers

  missing_dimers = np.random.randint(0, taper_length, size=(n_pf,))
  for pf, missing_n in zip(dimers, missing_dimers):
    if missing_n > 0:
      pf[-missing_n:] = 0
  return dimers


# pylint: disable=too-many-locals
def get_dimer_positions(dimers, show_progress=False):
  # Calculate the x, y and z positions of each dimers.

  n_pf = dimers.shape[0]
  n_rows = dimers.shape[1]

  # Get the parameters of geometry structure
  # according to the number of pf.
  params = get_structure_parameters().loc[n_pf].to_dict()

  long_dimer_distance = 8  # nm

  # Calculate the radius of the MT from n_pf
  mt_radius = ((16.4 * n_pf + 15) / 2) / 10

  # The dimer_factor is needed because we calculate theta from
  # dimer's center to dimer's center (not from monomer center).
  dimer_factor = 2/3

  # Calculate the theta angle (TODO: needs a better description)
  theta = (360 - n_pf * params['hrot']) * dimer_factor

  # Calculate the skew angle, the angle from one row to the next.
  skew_angle = -1 * np.arcsin(0.25 * mt_radius * np.sin(0.5 * np.deg2rad(theta)))

  # Init the list of positions
  #positions = pd.DataFrame()
  columns = ['row', 'pf', 'x', 'y', 'z', 'visible']
  n_columns = len(columns)
  positions = np.zeros((n_rows * n_pf, n_columns))

  # Calculate the position of the dimers
  # of the first row.
  i_row = 0
  for i_pf in range(n_pf):

    # Row index
    positions[i_pf, 0] = i_row

    # Protofilament index
    positions[i_pf, 1] = i_pf

    # Spatial coordinates (x, y, z)
    positions[i_pf, 2] = mt_radius * np.sin(i_pf * np.deg2rad(params['hrot']))
    positions[i_pf, 3] = mt_radius * np.cos(i_pf * np.deg2rad(params['hrot']))
    positions[i_pf, 4] = i_pf * params['htrans']

    # Is the dimer visible ?
    positions[i_pf, 5] = dimers[i_pf, i_row] == 1

  # Precompute shift for z coordinates.
  shifts = np.arange(1, n_rows) * long_dimer_distance * np.cos(skew_angle)

  # Precompute rotation angles.
  # pylint: disable=assignment-from-no-return
  rotations = np.deg2rad(np.arange(1, n_rows) * theta)

  # Starting from the first row, we calculcate
  # the rows of dimers above it by applying:
  # - a translation along the Z axis.
  # - a rotation on the Z axis
  first_row = positions[0:n_pf]
  for i_row in tqdm.trange(1, n_rows, leave=True, disable=not show_progress):

    current_row = first_row.copy()

    # Set the current row index
    current_row[:, 0] = i_row

    # Apply translation to z coordinates.
    current_row[:, 4] += shifts[i_row - 1]

    # Apply rotation
    rotation = rotations[i_row - 1]
    rot_z = Rotation.from_euler('z', rotation, degrees=False)
    current_row[:, 2:5] = rot_z.apply(current_row[:, 2:5])

    # Set dimer's visiblity
    current_row[:, 5] = dimers[:, i_row] == 1

    # Add new row's dimer positions to dataframe.
    array_index = i_row * n_pf
    positions[array_index:array_index + n_pf] = current_row

  positions = pd.DataFrame(positions, columns=columns)
  positions['visible'] = positions['visible'].astype('bool')
  return positions


def get_mt_tips(positions, coordinates_features=None):

  if coordinates_features is None:
    coordinates_features = ['x_pixel', 'y_pixel']

  # Get the position of the start and end of the microtubule
  indexed_positions = positions.set_index('row')
  indices = np.sort(indexed_positions.index.unique())

  first_dimers = indexed_positions.loc[indices[0], coordinates_features]
  x_start, y_start = first_dimers.mean()

  last_dimers = indexed_positions.loc[indices[-1], coordinates_features]
  x_end, y_end = last_dimers.mean()

  return x_start, x_end, y_start, y_end
