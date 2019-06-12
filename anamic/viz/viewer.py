import logging

import numpy as np
import pandas as pd
import panel as pn
import param
import bokeh as bk
from bokeh import plotting
import skimage
import matplotlib

from anamic.utils import css_dict_to_string


def reorder_image_dimensions(self, image, dimension_order=None):
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


def create_composite(images, colors):
  """Given a stack of 2D images and a list of colors. Create a composite
  RGB image.

  Args:
      images: array, a stack of 2D Numpy images.
      colors: list, a list of colors. The length of the list
        must be the same as `images`.

  Returns:
      composite, an RGB image.
  """

  if images.shape[0] != len(colors):
    mess = "The size of `colors` is different than the number of dimensions of `images`"
    raise ValueError(mess)

  colored_images = []
  for color, im in zip(colors, images):
    rgb = list(matplotlib.colors.to_rgb(color))
    im = skimage.img_as_float(im)
    im = skimage.color.gray2rgb(im)
    im *= rgb
    #im = skimage.color.rgb2hsv(im)
    colored_images.append(im)
  colored_images = np.array(colored_images)
  composite = np.sum(colored_images, axis=0)
  return composite


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
  COMPOSITE_COLORS = ['red', 'green', 'cyan', 'magenta', 'yellow']

  # Image parameters.
  time_param = param.Integer(default=0, bounds=(0, 10))
  z_param = param.Integer(default=0, bounds=(0, 1))
  channel_param = param.ObjectSelector()

  # Viewer parameters
  color_mode_param = param.ObjectSelector(default="Single", objects=["Single", "Composite"])
  colormap_param = param.ObjectSelector(default="Viridis256", objects=PALETTE_LIST)
  log_widget = pn.pane.Markdown("", css_classes=['log-widget'])

  def __init__(self, image, dimension_order=None, enable_log=True, **kwargs):
    super().__init__(**kwargs)

    self.image = reorder_image_dimensions(self, image, dimension_order=dimension_order)

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

    self.param.colormap_param.label = "Color Map"
    self.active_param_widgets.append('colormap_param')

    self.param.color_mode_param.label = "Color Mode"
    self.active_param_widgets.append('color_mode_param')

    # Assigne custom widget to certain params.
    self.param_widgets = {}
    self.param_widgets['channel_param'] = pn.widgets.RadioButtonGroup
    self.param_widgets['color_mode_param'] = pn.widgets.RadioButtonGroup

    # Create the Bokeh figure.
    self._create_figure()
    self._update_image_view()

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

    self.source = bk.models.ColumnDataSource(data={})

    figure_args = {}
    figure_args['tools'] = "pan,wheel_zoom,box_zoom,save,reset"
    figure_args['tooltips'] = [("x", "$x"), ("y", "$y"), ("value", "@image")]
    figure_args['active_scroll'] = "wheel_zoom"
    figure_args['match_aspect'] = True

    self.fig = plotting.figure(**figure_args)
    self.fig.toolbar.logo = None

    image_args = {}
    image_args['image'] = 'image'
    image_args['x'] = 'x'
    image_args['y'] = 'y'
    image_args['dw'] = 'dw'
    image_args['dh'] = 'dh'
    image_args['source'] = self.source

    if self.color_mode_param is "Composite":
      self.fig.image_rgba(**image_args)

    elif self.color_mode_param is "Single":
      self.color_mapper = bk.models.LinearColorMapper(palette=self.colormap_param)
      self.fig.image(color_mapper=self.color_mapper, **image_args)

      # Add colorbar
      color_bar = bk.models.ColorBar(color_mapper=self.color_mapper, location=(0, 0))
      self.fig.add_layout(color_bar, 'right')

    else:
      raise ValueError(f"Invalid color mode: {color_mode_param}")

    # Set figure aspect ratio and padding.
    self.fig.x_range.range_padding = 0
    self.fig.y_range.range_padding = 0
    self.fig.aspect_ratio = self.image_width / self.image_height

  def get_fig(self):
    return self.fig

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
    if self.color_mode_param is "Composite":

      # Create the composite image.
      # TODO: do we want to cache the entire image?
      images = self.image[self.time_param, :, self.z_param]
      colors = self.COMPOSITE_COLORS[:self.image_channel]
      frame = create_composite(images, colors)

      # Rescale to uint8 and add an alpha channel.
      frame = skimage.exposure.rescale_intensity(frame, out_range=(0, 255))
      frame = frame.astype('uint8')
      frame = np.insert(frame, 3, 255, axis=2)

    elif self.color_mode_param is "Single":
      channel_index = self.channel_names.index(self.channel_param)
      frame = self.image[self.time_param, channel_index, self.z_param]

    else:
      raise ValueError(f"Invalid color mode: {color_mode_param}")

    self._plot_frame(frame)

  @param.depends('color_mode_param', watch=True)
  def _change_color_mode(self):
    """Update viewer to match the new color mode.
    """
    self._create_figure()
    self._update_image_view()

  @param.depends('colormap_param', watch=True)
  def _update_colormap(self):
    """Update viewer to match the new colormap.

    TODO: ideally we won't have to recreate the entire image but only
    update the colormap. See https://github.com/bokeh/bokeh/issues/8991
    """
    self._create_figure()
    self._update_image_view()

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
    pane2 = pn.Row(pane1, self.get_fig, sizing_mode='stretch_both')

    if self.enable_log:
      log_container = pn.Column(self.log_widget, css_classes=['log-widget-container'], height=150)
      main = pn.Column(pane2, log_container)
    else:
      main = pn.Column(pane2)

    return main
