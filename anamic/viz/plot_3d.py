import numpy as np
import matplotlib


# pylint: disable=too-many-locals
def viz_dimer_positions(positions, size=5, cmap_name="tab20c", color_feature_name=None):

  try:
    import ipyvolume as ipv
  except ImportError:
    mess = "YOu need to install the ipyvolume library. "
    mess += "Please follow the official instructions at https://github.com/maartenbreddels/ipyvolume."
    raise ImportError(mess)

  # Only show visible dimers
  selected_dimers = positions[positions['visible']]

  x, y, z = selected_dimers[['x', 'y', 'z']].values.astype('float').T

  if color_feature_name:
    # TODO: that code should be much simpler...
    cmap = matplotlib.cm.get_cmap(cmap_name)
    categories = selected_dimers[color_feature_name].unique()
    color_indices = cmap([i / len(categories) for i in categories])

    color = np.zeros((len(selected_dimers[color_feature_name]), 4))
    for color_index, _ in enumerate(categories):
      mask = selected_dimers[color_feature_name] == categories[color_index]
      color[mask] = color_indices[color_index]
  else:
    color = '#e4191b'

  ipv.figure(height=800, width=1000)
  ipv.scatter(x, y, z, size=size, marker='sphere', color=color)
  ipv.squarelim()
  ipv.show()
