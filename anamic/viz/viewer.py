import logging

import numpy as np
import pandas as pd
import panel as pn
import param
import bokeh as bk
from bokeh import plotting

from anamic.utils import css_dict_to_string


def reorder_image_dimensions(image, dimension_order=None):
  """After wrapping the image in a new view, the shape of the image
  is modified to match the dimension order of 'TCZXY'.

  Args:
      dimension_order: str, the current order of dimension of the image. None
        assumes the dimensions to be 'XY', 'TXY', 'TCXY' or 'TCZXY'.

  Returns:
      The new view of the image.
  """
  dimension_orders = {}
  dimension_orders[2] = "XY"
  dimension_orders[3] = "TXY"
  dimension_orders[4] = "TCXY"
  dimension_orders[5] = "TCZXY"
  default_dimension_order = dimension_orders[5]

  image = image.view()

  if image.ndim < 2 or image.ndim > 5:
    mess = "The image needs to be of dimension 2 to 5. "
    mess += f"Number of dimension found is {image.ndim} of shape {image.shape}."
    raise ValueError(mess)

  if not dimension_order:
    current_order = dimension_orders[image.ndim]
  else:
    if len(dimension_order) != image.ndim:
      mess = "`dimension_order` needs to have the length of the image dimensions. "
      mess += f"dimension_order={len(dimension_order)} and image number of dimension is {image.ndim}."
      raise ValueError(mess)
    current_order = dimension_order
  current_order = list(current_order)

  missing_dims = set(default_dimension_order) - set(current_order)
  for missing_dim in missing_dims:
    image = np.expand_dims(image, axis=0)
    current_order.insert(0, missing_dim)

  new_order_indices = [current_order.index(dim) for dim in default_dimension_order]
  image = np.transpose(image, axes=new_order_indices)

  return image


# pylint: disable=too-many-instance-attributes
class ImageViewer(param.Parameterized):
  """An image viewer.

  Args:
      image: Numpy array. Shape can be from 5D to 2D.
      dimension_order: str, the current order of dimension of the image. None
        assumes the dimensions to be 'XY', 'TXY', 'TCXY' or 'TCZXY'.
      enable_log: bool, whether displaying a logging widget.
  """

  # pylint: disable=no-member
  PALETTE_LIST = bk.palettes.__palettes__

  # Image parameters.
  time_param = param.Integer(default=0, bounds=(0, 10))
  z_param = param.Integer(default=0, bounds=(0, 1))
  channel_param = param.ObjectSelector()

  # Viewer parameters
  colormap_param = param.ObjectSelector(default="Viridis256", objects=PALETTE_LIST)
  log_widget = pn.pane.Markdown("", css_classes=['log-widget'])

  def __init__(self, image, dimension_order=None, enable_log=True, **kwargs):
    super().__init__(**kwargs)

    self.image = reorder_image_dimensions(image, dimension_order=dimension_order)

    self._setup_logger()
    self._log_lines = []
    self.enable_log = enable_log

    # Get image dimensions.
    self.image_time = self.image.shape[0]
    self.image_channel = self.image.shape[1]
    self.image_z = self.image.shape[2]
    self.image_height = self.image.shape[3]
    self.image_width = self.image.shape[4]

    # Custom parameter widgets.
    self.active_param_widgets = []

    if self.image_time > 1:
      self.param.time_param.label = "Time"
      self.param.time_param.bounds = (0, self.image_time - 1)
      self.active_param_widgets.append('time_param')

    if self.image_z > 1:
      self.param.z_param.label = "Z"
      self.param.z_param.bounds = (0, self.image_time - 1)
      self.active_param_widgets.append('z_param')

    if self.image_channel > 1:
      self.param.channel_param.label = "Channel"
      self.channel_names = [f"Channel {i}" for i in range(self.image.shape[1])]
      self.param.channel_param.objects = self.channel_names
      self.param.set_default('channel_param', self.channel_names[0])
      self.active_param_widgets.append('channel_param')

    self.param.colormap_param.label = "Color Map (not working)"
    self.active_param_widgets.append('colormap_param')

    # Assigne custom widget to certain params.
    self.param_widgets = {}
    self.param_widgets['channel_param'] = pn.widgets.RadioButtonGroup

    # Create the Bokeh image.
    self.source = bk.models.ColumnDataSource(data={})

    # Create the Bokeh figure.
    self._create_figure()
    self._plot_frame(self.image[0, 0, 0])

    self.log.info("Image viewer has been correctly initialized.")

  def _setup_logger(self):
    self.log = logging.getLogger(f'ImageViewer-{str(self)}')
    self.log.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
    custom_handler = logging.StreamHandler()
    custom_handler.setFormatter(formatter)
    custom_handler.emit = self._logging_handler
    self.log.addHandler(custom_handler)

  def _logging_handler(self, record):
    """Log a message in the UI.
    """
    if self.enable_log:
      message = self.log.handlers[0].format(record)
      self._log_lines.append(message)
      self.log_widget.object = "<br/>".join(self._log_lines[::-1])

  def _set_css(self):
    """Define CSS properties.
    """
    css = {}
    css['.log-widget-container'] = {}
    css['.log-widget-container']['background'] = '#fbfbfb'
    css['.log-widget-container']['border'] = '1px #eaeaea solid !important'
    css['.log-widget-container']['overflow-y'] = 'auto'
    css['.log-widget-container']['min-width'] = '100%'
    css['.log-widget-container']['border-radius'] = '0'
    css['.log-widget-container']['padding'] = '25px !important'
    css['.log-widget-container']['resize'] = 'both'
    css['.log-widget'] = {}
    css['.log-widget']['min-width'] = '95%'

    css_string = css_dict_to_string(css)
    pn.extension(raw_css=[css_string])

  def _create_figure(self):
    """Create the Bokeh figure to display the image."""

    figure_args = {}
    figure_args['tools'] = "pan,wheel_zoom,box_zoom,save,reset"
    figure_args['tooltips'] = [("x", "$x"), ("y", "$y"), ("value", "@image")]
    figure_args['active_scroll'] = "wheel_zoom"
    figure_args['match_aspect'] = True

    self.fig = plotting.figure(**figure_args)
    self.fig.toolbar.logo = None

    self.color_mapper = bk.models.LinearColorMapper(palette=self.colormap_param)
    self.fig.image(image='image', x='x', y='y', dw='dw', dh='dh',
                   source=self.source, color_mapper=self.color_mapper)

    # Set figure aspect ratio and padding.
    self.fig.x_range.range_padding = 0
    self.fig.y_range.range_padding = 0
    self.fig.aspect_ratio = self.image_width / self.image_height

    # Add colorbar
    color_bar = bk.models.ColorBar(color_mapper=self.color_mapper, location=(0, 0))
    self.fig.add_layout(color_bar, 'right')

  def _plot_frame(self, frame):
    """Update the image Bokeh figure.

    Args:
        frame: a 2D array.
    """
    data = {}
    data['image'] = [frame]
    data['x'] = [0]
    data['y'] = [0]
    data['dw'] = [self.image_width]
    data['dh'] = [self.image_height]
    self.source.data = data

  @param.depends('time_param', 'channel_param', 'z_param', watch=True)
  def _update_image_view(self):
    """Callback that trigger the image update.
    """
    channel_index = self.channel_names.index(self.channel_param)
    frame = self.image[self.time_param, channel_index, self.z_param]
    self._plot_frame(frame)

  @param.depends('colormap_param', watch=True)
  def _update_colormap(self):
    """Callback that trigger the image update.
    """
    self.log.info(f"Update colormap to '{self.colormap_param}'.")

  def _get_image_info(self):
    """Get image informations about dimensions and current position
    in the viewer.

    Returns:
        infos: A Pandas dataframe converted to HTML code.
    """
    infos = []
    infos.append(('X', self.image_width, ""))
    infos.append(('Y', self.image_height, ""))
    if self.image_z > 1:
      infos.append(('Z', self.image_z, self.z_param))
    if self.image_time > 1:
      infos.append(('Time', self.image_time, self.time_param))
    if self.image_channel > 1:
      infos.append(('Channel', self.image_channel, self.channel_names.index(self.channel_param)))
    infos = pd.DataFrame(infos, columns=['Dimension', 'Size', 'Position'])
    return infos.to_html(index=False)

  def panel(self):
    """The image viewer as a panel to display.
    """

    self._set_css()

    # Widget to show image informations.
    info_widget = self._get_image_info

    # Widget with parameter widgets to control image viewer.
    parameters_widget = pn.Param(self.param, parameters=self.active_param_widgets, widgets=self.param_widgets)

    # Organize the widgets and containers to the final UI.
    pane1 = pn.Column(info_widget, parameters_widget)
    pane2 = pn.Row(pane1, self.fig, sizing_mode='stretch_both')

    if self.enable_log:
      log_container = pn.Column(self.log_widget, css_classes=['log-widget-container'], height=100)
      main = pn.Column(pane2, log_container)
    else:
      main = pn.Column(pane2)

    return main
