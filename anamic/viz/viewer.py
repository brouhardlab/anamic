import numpy as np
import pandas as pd
import panel as pn
import param
import bokeh as bk
from bokeh import plotting
import skimage

from anamic.utils import css_dict_to_string
from anamic.imaging import reorder_image_dimensions
from anamic.imaging import create_composite
from anamic.imaging import get_palettes
from .log_widget import LoggingWidget


# pylint: disable=too-many-instance-attributes
class ImageViewer(param.Parameterized):
  """An image viewer.

  Args:
      image: Numpy array. Shape can be from 5D to 2D.
      dimension_order: str, the current order of dimension of the image. None
        assumes the dimensions to be 'XY', 'TXY', 'TCXY' or 'TCZXY'.
      enable_log: bool, whether displaying a logging widget.
  """

  COMPOSITE_COLORS = ['red', 'green', 'cyan', 'magenta', 'yellow']

  # Image parameters.
  time_param = param.Integer(default=0, bounds=(0, 10))
  z_param = param.Integer(default=0, bounds=(0, 1))
  channel_param = param.ObjectSelector()

  # Viewer parameters
  color_mode_param = param.ObjectSelector(default="Single", objects=["Single", "Composite"])
  colormap_param = param.ObjectSelector()
  log_widget = pn.pane.Markdown("", css_classes=[])

  def __init__(self, image, dimension_order=None, enable_log=True, **kwargs):
    super().__init__(**kwargs)

    # Reshape the image
    self.image = reorder_image_dimensions(image, dimension_order=dimension_order)

    # Get an ID for this viewer's instance
    self.viewer_id = id(self)

    # Setup the logger
    self.log = LoggingWidget(logger_name=f"Imageviewer-{self.viewer_id}", enable=enable_log)

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
      self.param.z_param.bounds = (0, self.image_z - 1)
      self.active_param_widgets.append('z_param')

    if self.image_channel > 1:
      self.param.channel_param.label = "Channel"
      self.channel_names = [f"Channel {i}" for i in range(self.image.shape[1])]
      self.param.channel_param.objects = self.channel_names
      self.param.set_default('channel_param', self.channel_names[0])
      self.active_param_widgets.append('channel_param')

    self.param.colormap_param.label = "Color Map"
    self.active_param_widgets.append('colormap_param')
    self.palettes = get_palettes()
    self.param.colormap_param.objects = [name.capitalize() for name in self.palettes.keys()]
    self.param.set_default('colormap_param', "Viridis")

    self.param.color_mode_param.label = "Color Mode"
    self.active_param_widgets.append('color_mode_param')

    # Assigne custom widget to certain params.
    self.param_widgets = {}
    self.param_widgets['channel_param'] = pn.widgets.RadioButtonGroup
    self.param_widgets['color_mode_param'] = pn.widgets.RadioButtonGroup

    # Create the Bokeh figure.
    self.fig = None
    self._create_figure()
    self._update_image_view()

    self.log.info("Image viewer has been correctly initialized.")

  def _set_css(self):
    """Define CSS properties.
    """
    css = {}
    css[f'.viewer-{self.viewer_id}'] = {}
    # css[f'.viewer-{self.viewer_id}']['border'] = '1px #5d5d5d solid !important'
    #css[f'.viewer-{self.viewer_id}']['min-width'] = '100% !important'
    #css[f'.viewer-{self.viewer_id}']['min-height'] = '100% !important'

    css_string = css_dict_to_string(css)
    pn.extension(raw_css=[css_string])

  def _create_figure(self):
    """Create the Bokeh figure to display the image."""

    self.source = bk.models.ColumnDataSource(data={})

    image_tooltips = []
    image_tooltips.append(('x, y', '$x{0,0.00}, $y{0,0.00}'))
    if self.image_time > 1:
      image_tooltips.append(('Time', '@time'))
    if self.image_z > 1:
      image_tooltips.append(('Z', '@z'))
    if self.image_channel > 1:
      image_tooltips.append(('Channel', '@channel'))

    values = ", ".join([f'@channel_{i}' for i in range(self.image_channel)])
    image_tooltips.append(('Values', values))

    figure_args = {}
    figure_args['tools'] = "pan,wheel_zoom,box_zoom,save,reset"
    figure_args['tooltips'] = image_tooltips
    figure_args['active_scroll'] = "wheel_zoom"
    figure_args['match_aspect'] = True
    figure_args['sizing_mode'] = 'stretch_both'

    if self.fig:
      figure_args['x_range'] = self.fig.x_range
      figure_args['y_range'] = self.fig.y_range

    self.fig = plotting.figure(**figure_args)
    self.fig.toolbar.logo = None

    image_args = {}
    image_args['image'] = 'image'
    image_args['x'] = 'x'
    image_args['y'] = 'y'
    image_args['dw'] = 'dw'
    image_args['dh'] = 'dh'
    image_args['source'] = self.source

    if self.color_mode_param == "Composite":
      self.fig.image_rgba(**image_args)
      self.color_bar = None

    elif self.color_mode_param == "Single":
      palette = self.palettes[str(self.colormap_param).lower()]
      self.color_mapper = bk.models.LinearColorMapper(palette=palette)
      self.fig.image(color_mapper=self.color_mapper, **image_args)

      # Add colorbar
      self.color_bar = bk.models.ColorBar(color_mapper=self.color_mapper, location=(0, 0))

    else:
      raise ValueError(f"Invalid color mode: {self.color_mode_param}")

    # Set figure aspect ratio and padding.
    self.fig.x_range.range_padding = 0
    self.fig.y_range.range_padding = 0
    self.fig.aspect_ratio = self.image_width / self.image_height

  def _get_fig(self):
    return self.fig

  def _plot_frame(self, frame, metadata):
    """Update the image Bokeh figure.

    Args:
        frame: a 2D array.
        metadata: dict, used for the tooltips.
    """
    data = {}
    data['image'] = [frame]
    data['x'] = [0]
    data['y'] = [0]
    data['dw'] = [self.image_width]
    data['dh'] = [self.image_height]
    data.update(metadata)
    self.source.data = data

  @param.depends('time_param', 'channel_param', 'z_param', watch=True)
  def _update_image_view(self):
    """Callback that trigger the image update.
    """
    channel_index = self.channel_names.index(self.channel_param)

    if self.color_mode_param == "Composite":

      # Create the composite image.
      # TODO: do we want to cache the entire image?
      images = self.image[self.time_param, :, self.z_param]
      colors = self.COMPOSITE_COLORS[:self.image_channel]
      frame = create_composite(images, colors)

      # Rescale to uint8 and add an alpha channel.
      frame = skimage.exposure.rescale_intensity(frame, out_range=(0, 255))
      frame = frame.astype('uint8')
      frame = np.insert(frame, 3, 255, axis=2)

    elif self.color_mode_param == "Single":
      frame = self.image[self.time_param, channel_index, self.z_param]
      images = frame

    else:
      raise ValueError(f"Invalid color mode: {self.color_mode_param}")

    metadata = {}
    if self.image_time > 1:
      metadata['time'] = [self.time_param]
    if self.image_z > 1:
      metadata['z'] = [self.z_param]
    if self.image_channel > 1:
      metadata['channel'] = [self.channel_param]

    for i in range(self.image_channel):
      metadata[f'channel_{i}'] = [self.image[self.time_param, i, self.z_param]]

    self._plot_frame(frame, metadata)

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

  def _get_color_bar(self):
    # if self.color_bar:
    #   cbar_fig = plotting.figure(tools="",  toolbar_location=None, min_border=0, outline_line_color=None, height=self.fig.height)
    #   cbar_fig.add_layout(self.color_bar)
    #   bar_container = pn.Row(cbar_fig, width_policy='min', height_policy='min')
    #   return None
    # else:
    #   return None
    return None

  def panel(self):
    """The image viewer as a panel to display.
    """

    self._set_css()

    # Widget to show image informations.
    info_widget = self._get_image_info

    # Widget with parameter widgets to control image viewer.
    parameters_widget = pn.Param(self.param, parameters=self.active_param_widgets, widgets=self.param_widgets)

    # Organize the widgets and containers to the final UI.
    tool_widget = pn.Column(info_widget, parameters_widget)
    image_container = pn.Row(self._get_fig, self._get_color_bar, sizing_mode='scale_both')

    content_container = pn.Row(tool_widget, image_container)
    main = pn.Column(content_container, self.log.panel(), css_classes=[f'viewer-{self.viewer_id}'], sizing_mode='scale_both')
    return main
