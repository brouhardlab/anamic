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


def get_default_values(name):

  default_values = {}
  default_values['line_alpha'] = 1
  default_values['line_color'] = 'black'
  default_values['line_width'] = 1

  default_values['fill_alpha'] = 1
  default_values['fill_color'] = 'black'

  default_values['size'] = 10
  default_values['radius'] = None
  default_values['angle'] = 0

  if name not in default_values.keys():
    return None

  return default_values[name]


# pylint: disable=too-many-instance-attributes
class ObjectDrawer(param.Parameterized):
  """Draw objects on a Bokeh figure.

  Args:
    fig: Bokeh Figure.
    log: Python logger
  """

  def __init__(self, fig, fig_pane, log, **kwargs):
    super().__init__(**kwargs)
    self.fig = fig
    self.fig_pane = fig_pane
    self.log = log

    self.renderers = {}
    self.data = {}
    self.source = []
    self.hover_tool = None

    self.cursor = {}
    self.cursor['time_index'] = 0
    self.cursor['channel_index'] = 0
    self.cursor['z_index'] = 0

    self._init_glyph()

  def _init_glyph(self):
    self._init_markers()

  def _init_markers(self):
    for name, mark_obj in get_markers().items():
      # Create glyph object.
      args = {arg: arg for arg in mark_obj.dataspecs()}
      glyph = mark_obj(**args)

      # Create source data with empty vectors.
      empty_data = {arg: [] for arg in mark_obj.dataspecs()}
      source = bk.models.ColumnDataSource(empty_data)

      # Create the filter.
      index_filter = bk.models.IndexFilter([])
      view = bk.models.CDSView(source=source, filters=[index_filter])

      # Add glyph to the figure.
      renderer = self.fig.add_glyph(source, glyph, view=view)

      self.renderers[name] = renderer
      self.data[name] = pd.DataFrame()

    # Add the associated tooltips.
    if self.hover_tool and self.hover_tool in self.fig.tools:
      self.fig.tools.remove(self.hover_tool)

    tooltips = [('x', '@x'), ('y', '@y')]
    renderers = list(self.renderers.values())
    self.hover_tool = bk.models.tools.HoverTool(tooltips=tooltips, renderers=renderers)
    self.fig.add_tools(self.hover_tool)

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

  def add_markers(self, datum, marker='circle', log=True):
    self.data[marker] = pd.concat([self.data[marker], datum], ignore_index=True, sort=False)

    source = self.renderers[marker].data_source

    # Get the length of the current source data.
    if source.column_names:
      n = len(source.data[source.column_names[0]])
    else:
      n = 0

    datum = datum.reset_index()

    # Add missing columns to the source data.
    for col in datum.columns:
      if col not in source.column_names:
        # For the exisitng rows we set the default values
        # to None. In the future, we might want
        # to have a list of defined default
        # values according to the name of the column.
        values = [get_default_values(col)] * n
        source.add(values, col)

    # Before streaming, we add missing columns with
    # default values to the new data.
    for col in source.column_names:
      if col not in datum.columns:
        datum[col] = [get_default_values(col)] * datum.shape[0]

    # Add the new data to the source.
    source.stream(datum)

    # Draw markers.
    self.draw_markers()

    if datum.shape[0] > 0 and log:
      self.log.info(f'{datum.shape[0]} marker(s) of type "{marker}" have been added.')

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
      index_filter = bk.models.IndexFilter(list(data_view.index.values))
      renderer.view.filters[0] = index_filter

    try:
      self.fig_pane.param.trigger('object')
    except KeyError:
      pass

  def _clear_marker(self, name):
    mark_obj = get_markers()[name]

    # Clear data source.
    empty_data = {arg: [] for arg in mark_obj.dataspecs()}
    source = bk.models.ColumnDataSource(empty_data)
    self.renderers[name].data_source = source

    # Reset data filter
    index_filter = bk.models.IndexFilter([])
    self.renderers[name].view.filters[0] = index_filter

    self.data[name] = pd.DataFrame()

  def clear(self, name=None):
    """Clear all glyphs. If name is provided only remove glyphs for this type."""
    if name:
      self._clear_marker(name)
    else:
      for object_name in self.renderers.keys():
        self._clear_marker(object_name)
    self.draw_markers()

  def update_fig(self, fig, fig_pane):
    """Call this method whenever the figure is a new instance.
    """
    self.fig = fig
    self.fig_pane = fig_pane
    new_renderers = {}
    for name, renderer in self.renderers.items():
      renderer = self.fig.add_glyph(renderer.data_source, renderer.glyph, view=renderer.view)
      new_renderers[name] = renderer
    self.renderers = new_renderers

  def panel(self):
    source = self.renderers['circle'].data_source
    table_widget = bk.models.widgets.DataTable(source=source)
    return table_widget
