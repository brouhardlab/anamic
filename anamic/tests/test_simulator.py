from pathlib import Path

import numpy as np
import numpy.testing as npt
import anamic

root_dir = Path(__file__).parent
data_dir = root_dir / "data"


def test_get_dimer_positions():
    n_pf = 13
    n_rows = 100
    dimers = np.ones((n_pf, n_rows))
    positions = anamic.simulator.get_dimer_positions(dimers)

    # Check row and pf columns.
    npt.assert_array_equal(positions['row'].unique(), np.arange(0, n_rows))
    npt.assert_array_equal(positions['pf'].unique(), np.arange(0, n_pf))

    # Check the first 10 `x` values.
    x_values = [0, 5.30208447, 9.38972376, 11.32664199, 10.66918704, 7.5679493,
                2.73326888, -2.72746819, -7.56347726, -10.66706797]
    npt.assert_almost_equal(positions['x'].iloc[:10].values, x_values, decimal=8)

    # Check the first 10 `y` values.
    y_values = np.array([11.41, 10.10326681, 6.48237516, 1.37669214, -4.04432293,
                        -8.53898374, -11.07778594, -11.07921555, -8.54294514, -4.04990875])
    npt.assert_almost_equal(positions['y'].iloc[:10].values, y_values, decimal=8)

    # Check the first 10 `z` values.
    z_values = np.array([0, 0.939, 1.878, 2.817, 3.756, 4.695, 5.634, 6.573, 7.512, 8.451])
    npt.assert_almost_equal(positions['z'].iloc[:10].values, z_values, decimal=8)

    assert positions['visible'].sum() == n_pf * n_rows
    assert positions.shape == (n_pf * n_rows, 6)


def test_get_structure_parameters():
    params = anamic.simulator.get_structure_parameters()
    assert params.shape == (16, 3)
    npt.assert_array_equal(params.mean().values, [1.25, 10.1275, 0.377125])


def test_dimers_builder():
    n_pf = 13
    mt_length_nm = 2000
    taper_length_nm = 200
    dimers = anamic.simulator.dimers_builder(n_pf, mt_length_nm, taper_length_nm)

    # Check the shape of the dimers' array
    n_rows = mt_length_nm / 8
    assert dimers.shape == (n_pf, n_rows)

    # Check dimers' count.
    assert dimers.sum() < (n_rows * n_pf)

    # Minimum possible number of dimers
    min_dimers = n_rows * n_pf * (1 - taper_length_nm / mt_length_nm)
    assert dimers.sum() > min_dimers
