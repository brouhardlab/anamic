from pathlib import Path

import matplotlib.pyplot as plt
import ipyvolume as ipv
import pandas as pd
import numpy as np


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