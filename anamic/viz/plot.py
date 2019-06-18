import numpy as np
import matplotlib.pyplot as plt

from anamic.simulator import structure


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


def viz_2d_dimers_positions(ax, dimers, x_feature, y_feature, color_feature, marker_size, x_offset=None, colormap='tab20c'):
  colors = dimers[color_feature].values
  ax.scatter(dimers[x_feature], dimers[y_feature], c=colors, s=marker_size, cmap=colormap)
  ax.set_aspect('equal')
  if x_offset:
    ax.set_xlim(dimers[x_feature].max() - x_offset, dimers[x_feature].max() + 5)
  else:
    ax.set_xlim(dimers[x_feature].min() - 5, dimers[x_feature].max() + 5)
  ax.set_ylim(dimers[y_feature].min() - 5, dimers[y_feature].max() + 5)


def imshow_colorbar(im, ax):
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  img = ax.imshow(im, interpolation='none', origin=[0, 0])
  ax.set_aspect('equal')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img, cax=cax)


def show_tips(ax, positions, coordinates_features, marker_size=80):
  x_start, x_end, y_start, y_end = structure.get_mt_tips(positions, coordinates_features)
  ax.scatter([x_start], [y_start], color='cyan', label='Start', marker='x', s=marker_size)
  ax.scatter([x_end], [y_end], color='red', label='End', marker='x', s=marker_size)
  ax.legend()
