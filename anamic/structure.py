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


def get_long_dimer_spacing():
    """In angstrom"""
    return 80


def get_radius(n_pf):
    """From Chretien, 1991"""
    return ((16.4 * n_pf + 46.8) / 2)


def generate_random_tapers(dimers, min_dimer, max_dimer):
    n_pf = dimers.shape[0]
    missing_dimers = np.random.randint(min_dimer, max_dimer, size=(n_pf,))
    for pf, missing_n in zip(dimers, missing_dimers):
        if missing_n > 0:
            pf[-missing_n:] = 0
    return dimers


def get_dimer_positions(dimers):
    # Calculate dimer positions
    n_pf = dimers.shape[0]
    n_rows = dimers.shape[1]

    # Get the parameters of geometry structure
    # according to the number of pf.
    params = get_structure_parameters().loc[n_pf].to_dict()

    mt_radius = ((16.4 * n_pf + 46.8) / 2) / 10
    # Equation 6 of Chretien, 1996
    skew_angle = - np.arctan((3 * 2 / (13 * 51.5)) - (params['helix_start_number'] * 2 / ( n_pf * 51.5)))

    positions = []

    i_row = 0
    for i_pf in range(n_pf):
        datum = {}
        datum['row'] = i_row
        datum['pf'] = i_pf
        datum['helix'] = 0
        datum['x'] = mt_radius * np.sin(i_pf * np.deg2rad(params['hrot']))
        datum['y'] = mt_radius * np.cos(i_pf * np.deg2rad(params['hrot']))
        datum['z'] = i_pf * params['htrans']
        datum['visible'] = False
        datum['visible'] = dimers[i_pf, i_row] == 1
        positions.append(datum)
    positions = pd.DataFrame(positions)

    for i_row in range(1, n_rows):

        initial_row = positions[positions['row'] == 0]
        current_row = initial_row.copy()

        current_row['row'] = i_row

        # Apply translation.
        shift = i_row * 8 * np.cos(skew_angle)
        current_row['z'] += shift

        # Apply rotation
        rotation = i_row * (360 - n_pf * params['hrot']) / params['helix_start_number'] * 2
        rotation = np.deg2rad(rotation)
        Rz = transformations.rotation_matrix(rotation, [0, 0, 1])
        current_row[['x', 'y', 'z']] = np.dot(current_row[['x', 'y', 'z']].values, Rz[:3, :3].T)

        current_row['visible'] = False
        current_row['visible'] = dimers[:, i_row] == 1

        # Add new positions to the list
        positions = positions.append(current_row, ignore_index=True)
    return positions