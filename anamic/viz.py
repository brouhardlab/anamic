import matplotlib.pyplot as plt
import ipyvolume as ipv
import numpy as np


def viz_dimers(dimers, start_row=0, grid=True):
    fig, ax = plt.subplots(figsize=(30, 20))
    ax.imshow(dimers, interpolation='none', aspect='equal', cmap='tab10')
    if grid:
        ax.set_xticks(np.arange(start_row - 0.5, dimers.shape[1] - 0.5), minor=True)
        ax.set_yticks(np.arange(-0.5, dimers.shape[0] - 0.5), minor=True)
        ax.grid(which='minor', color='black', alpha=0.4)

        ax.set_xlim(start_row - 0.5, dimers.shape[1] - 0.5)
        ax.set_ylim(-0.5, dimers.shape[0] - 0.5)
    return fig


def viz_dimer_positions(positions, size=5, use_ipv=True):
    x, y, z = positions[['x', 'y', 'z']].values.astype('float').T

    if use_ipv:
        ipv.figure(height=400, width=800)
        ipv.quickscatter(x, y, z, size=size, marker='sphere', color='#e4191b')
        ipv.squarelim()
        ipv.show()
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='#e4191b', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
