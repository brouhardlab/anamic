import pandas as pd
import param
import bokeh as bk

def get_markers():
  markers = {}
  markers['asterisk'] = bk.models.markers.Asterisk
  markers['circle'] = bk.models.markers.Circle
  markers['circleCross'] = bk.models.markers.CircleCross
  markers['circleX'] = bk.models.markers.CircleX
  markers['cross'] = bk.models.markers.Cross
  markers['cash'] = bk.models.markers.Dash
  markers['diamond'] = bk.models.markers.Diamond
  markers['diamondCross'] = bk.models.markers.DiamondCross
  markers['hex'] = bk.models.markers.Hex
  markers['invertedTriangle'] = bk.models.markers.InvertedTriangle
  markers['square'] = bk.models.markers.Square
  markers['squareCross'] = bk.models.markers.SquareCross
  markers['squareX'] = bk.models.markers.SquareX
  markers['riangle'] = bk.models.markers.Triangle
  markers['x'] = bk.models.markers.X
  return markers


class ObjectDrawer(param.Parameterized):
  """Draw objects on a Bokeh figure.

  Args:
    fig: Bokeh Figure.
    log: Python logger
  """

  def __init__(self, fig, log, **kwargs):
    super().__init__(**kwargs)
    self.fig = fig
    self.log = log

    self.renderers = {}
    self.data = {}

    self.cursor = {}
    self.cursor['time_index'] = 0
    self.cursor['channel_index'] = 0
    self.cursor['z_index'] = 0

  def _clear_columns(self, data):
    """Remove uneeded columns of a DataFrame when feeding to a
    Bokeh glyph.
    """
    if not isinstance(data, pd.DataFrame):
      raise TypeError(f"`data` needs to be a Pandas DataFrame not: {type(data)}")

    to_remove = ['time_index', 'channel_index', 'z_index']
    to_keep = []
    for name in data.columns:
      if name not in to_remove:
        to_keep.append(name)
    return to_keep

  def draw(self):
    self.draw_markers()

  def add_markers(self, datum, marker='circle'):
    marker_obj = get_markers()[marker]

    # Create a new data source and associated glyph.
    if marker not in self.renderers.keys():
      self.data[marker] = datum

      # Create a new glyph for this marker.
      glyph_args = {name: name for name in self._clear_columns(datum)}
      glyph = marker_obj(**glyph_args)

      source = bk.models.ColumnDataSource(datum)
      index_filter = bk.models.IndexFilter([])
      view = bk.models.CDSView(source=source, filters=[index_filter])
      self.renderers[marker] = self.fig.add_glyph(source, glyph, view=view)

    # Add new data to the existing source.
    else:
      self.data[marker] = pd.concat([self.data[marker], datum], ignore_index=True)
      self.renderers[marker].data_source.stream(datum)

    self.draw_markers()

  def draw_markers(self):
    """Draw markers for the current cursor."""

    for name in self.renderers.keys():
      renderer = self.renderers[name]
      datum = self.data[name]

      # Filter the data to draw.
      data_view = datum
      for axis_name, idx in self.cursor.items():
        # Don't filter if cursor is set to None.
        if axis_name in datum.columns and idx is not None:
          # Keep data equals to current index and data set to None.
          mask = (data_view[axis_name] == idx) | (pd.isnull(data_view[axis_name]))
          data_view = data_view[mask]

      # Replace the indices of the filter for the markers to draw.
      renderer.view.filters[0].indices = list(data_view.index.values)

  def clear(self, name=None):
    """Clear all glyphs. If name is provided only remove glyphs for this type."""
    if name:
      self.fig.renderers.remove(self.renderers[name])
      self.renderers.pop(name)
      self.data.pop(name)
    else:
      for _, renderer in self.renderers.items():
        if renderer in self.fig.renderers:
          self.fig.renderers.remove(renderer)
      self.renderers = {}
      self.data = {}

  def update_fig(self, fig):
    """Call this method whenever the figure is a new instance.
    """
    self.fig = fig
    data = self.data.copy()
    self.clear()
    for name, datum in data.items():
      self.add_markers(datum, name)
