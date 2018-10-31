from pathlib import Path

import matplotlib.pyplot as plt
import ipyvolume as ipv
import pandas as pd
import numpy as np

from . import transformations

DATA_DIR = Path(__file__).parent / "data"


def get_structure_parameters():
    struct = pd.read_csv(DATA_DIR / 'mt_structure.csv')
    struct = struct.set_index('n_pf', drop=True)
    return struct


def generate_random_tapers(dimers, min_dimer, max_dimer):
    n_pf = dimers.shape[0]
    missing_dimers = np.random.randint(min_dimer, max_dimer, size=(n_pf,))
    for pf, missing_n in zip(dimers, missing_dimers):
        if missing_n > 0:
            pf[-missing_n:] = 0
    return dimers


def generate_uniform_taper(dimers, taper_length_nm):
    long_dimer_distance = 8  # nm
    taper_length = int(np.round(taper_length_nm) / long_dimer_distance)
    n_pf = dimers.shape[0]
    missing_dimers = np.random.randint(0, taper_length, size=(n_pf,))
    for pf, missing_n in zip(dimers, missing_dimers):
        if missing_n > 0:
            pf[-missing_n:] = 0
    return dimers


def get_dimer_positions(dimers):
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
    positions = pd.DataFrame()

    # Calculate the position of the dimers
    # of the first row.
    i_row = 0
    for i_pf in range(n_pf):
        datum = {}

        # Row index
        datum['row'] = i_row

        # Protofilament index
        datum['pf'] = i_pf

        # Spatial coordinates
        datum['x'] = mt_radius * np.sin(i_pf * np.deg2rad(params['hrot']))
        datum['y'] = mt_radius * np.cos(i_pf * np.deg2rad(params['hrot']))
        datum['z'] = i_pf * params['htrans']

        # Is the dimer visible ?
        datum['visible'] = dimers[i_pf, i_row] == 1

        # Store the positions in the dataframe
        positions = positions.append(pd.Series(datum), ignore_index=True)

    # Starting from the first row, we calculcate
    # the rows of dimers above it by applying:
    # - a translation along the Z axis.
    # - a rotation on the Z axis
    i_helix = 0
    first_row = positions[positions['row'] == 0]
    for i_row in range(1, n_rows):

        current_row = first_row.copy()
        current_row['row'] = i_row

        # Apply translation.
        shift = i_row * long_dimer_distance * np.cos(skew_angle)
        current_row['z'] += shift

        # Apply rotation
        rotation = np.deg2rad(i_row * theta)
        Rz = transformations.rotation_matrix(rotation, [0, 0, 1])
        current_row[['x', 'y', 'z']] = np.dot(current_row[['x', 'y', 'z']].values, Rz[:3, :3].T)
        current_row['visible'] = dimers[:, i_row] == 1

        # Add new row's dimer positions to dataframe.
        positions = positions.append(current_row, ignore_index=True)
        
    return positions