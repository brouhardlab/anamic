import matplotlib
import matplotlib.pyplot as plt
import ipyvolume as ipv
import numpy as np


def viz_dimers(dimers, start_row=0, grid=True):
    size_ratio = dimers.shape[1] / dimers.shape[0]
    size = int(size_ratio * 3)
    size = max(size, 10)

    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(dimers, interpolation='none', aspect='equal', cmap='tab10')
    if grid:
        ax.set_xticks(np.arange(start_row - 0.5, dimers.shape[1] - 0.5), minor=True)
        ax.set_yticks(np.arange(-0.5, dimers.shape[0] - 0.5), minor=True)
        ax.grid(which='minor', color='black', alpha=0.4)

        ax.set_xlim(start_row - 0.5, dimers.shape[1] - 0.5)
        ax.set_ylim(-0.5, dimers.shape[0] - 0.5)
    return fig


def viz_dimer_positions(positions, size=5, color_feature_name=None):
    # Only show visible dimers
    selected_dimers = positions[positions['visible'] == True]

    x, y, z = selected_dimers[['x', 'y', 'z']].values.astype('float').T

    if color_feature_name:
        cmap = matplotlib.cm.get_cmap('tab20c')
        color = cmap(selected_dimers[color_feature_name].values)
    else:
        color = '#e4191b'

    ipv.figure(height=800, width=1000)
    ipv.scatter(x, y, z, size=size, marker='sphere', color=color)
    ipv.squarelim()
    ipv.show()
