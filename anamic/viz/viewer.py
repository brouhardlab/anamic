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
from .drawer import ObjectDrawer


# pylint: disable=too-many-instance-attributes
class ImageViewer(param.Parameterized):
  """An image viewer.

  Args:
      image: Numpy array. Shape can be from 5D to 2D.
      dimension_order: str, the current order of dimension of the image. None
        assumes the dimensions to be 'XY', 'TXY', 'TCXY' or 'TCZXY'.
      enable_log: bool, whether displaying a logging widget.
      width: int, width of the viewer.
      height: int, height of the viewer.
  """

  COMPOSITE_COLORS = ['red', 'green', 'cyan', 'magenta', 'yellow']

  # Image parameters.
  time_param = param.Integer(default=0, bounds=(0, 10))
  z_param = param.Integer(default=0, bounds=(0, 1))
  channel_param = param.ObjectSelector()

  # Viewer parameters
  intensities_param = param.Range(default=(3, 7), bounds=(0, 10))
  color_mode_param = param.ObjectSelector(default="Single", objects=["Single", "Composite"])
  colormap_param = param.ObjectSelector()

  # pylint: disable=too-many-statements
  def __init__(self, image, dimension_order=None, enable_log=True, width=None,
               height=None, _drawer_class=ObjectDrawer, **kwargs):
    super().__init__(**kwargs)

    # Get an ID for this viewer's instance
    self.viewer_id = id(self)

    # Setup the logger
    self.log = LoggingWidget(logger_name=f"Imageviewer-{self.viewer_id}", enable=enable_log)

    if image.dtype.name == 'float64':
      image = image.astype('float32')
      self.log.warning("Image of type 'float64' have been converted to 'float32'")

    self.width = width
    self.height = height

    # Reshape the image
    self.image = reorder_image_dimensions(image, dimension_order=dimension_order)

    # Init attributes
    self.fig = None
    self.fig_pane = None
    self.image_renderer = None
    self.color_bar = None
    self.color_mapper = None
    self.image_hover_tool = None
    self.drawer = None

    # Get image informations.
    self.image_time = self.image.shape[0]
    self.image_channel = self.image.shape[1]
    self.image_z = self.image.shape[2]
    self.image_height = self.image.shape[3]
    self.image_width = self.image.shape[4]

    # Store min and max of the image per channel.
    self.image_max = [self.image[:, i].max() for i in range(self.image_channel)]
    self.image_min = [self.image[:, i].min() for i in range(self.image_channel)]

    # Set intensities bounds.
    self.intensities_bounds = [(self.image_min[i], self.image_max[i]) for i in range(self.image_channel)]

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

    self.param.intensities_param.label = "Intensity Range"
    self.active_param_widgets.append('intensities_param')
    if self.image.dtype.kind == 'f':
      bounds = np.finfo(self.image.dtype).min, np.finfo(self.image.dtype).max
    else:
      bounds = np.iinfo(self.image.dtype).min, np.iinfo(self.image.dtype).max
    self.param.intensities_param.bounds = bounds
    self.intensities_param = self.intensities_bounds[self._get_channel_index()]

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
    self._init_figure()
    self._plot_image()

    # Init the object drawer
    if _drawer_class:
      self.drawer = _drawer_class(self.fig, self.fig_pane, self.log)

    # Trigger first plot update.
    self._update_image_view()

    # Update range for values.
    self._update_intensities_slider_bounds()

    self.log.info("Image viewer has been correctly initialized.")

  def _get_channel_index(self):
    if self.image_channel > 1:
      return self.channel_names.index(self.channel_param)
    return 0

  def _set_css(self):
    """Define CSS properties.
    """
    css = {}
    css[f'.viewer-{self.viewer_id}'] = {}
    # css[f'.viewer-{self.viewer_id}']['border'] = '1px #9d9d9d solid !important'
    css_string = css_dict_to_string(css)
    pn.extension(raw_css=[css_string])

  def _init_figure(self):
    """Init the Bokeh figure object
    """
    self.source = bk.models.ColumnDataSource(data={})

    figure_args = {}
    figure_args['tools'] = ["pan", "wheel_zoom", "box_zoom", "save",
                            "zoom_in", "zoom_out", "reset"]
    figure_args['active_scroll'] = "wheel_zoom"
    figure_args['match_aspect'] = True
    figure_args['sizing_mode'] = 'stretch_both'

    if self.fig:
      figure_args['x_range'] = self.fig.x_range
      figure_args['y_range'] = self.fig.y_range

    self.fig = plotting.figure(**figure_args)
    self.fig_pane = pn.pane.Bokeh(self.fig)

    # Configure the figure.
    self.fig.toolbar.logo = None
    self.fig.tools[2].match_aspect = True

    # Set figure aspect ratio and padding.
    self.fig.x_range.range_padding = 0
    self.fig.y_range.range_padding = 0
    self.fig.aspect_ratio = self.image_width / self.image_height

  def _plot_image(self):
    """Plot the current image and remove the previous one if it exists.
    """

    if self.fig:
      if self.image_renderer and self.image_renderer in self.fig.renderers:
        self.fig.renderers.remove(self.image_renderer)
      if self.color_bar and self.color_bar in self.fig.renderers:
        self.fig.renderers.remove(self.color_bar)
      if self.image_hover_tool and self.image_hover_tool in self.fig.tools:
        self.fig.tools.remove(self.image_hover_tool)

    image_args = {}
    image_args['image'] = 'image'
    image_args['x'] = 'x'
    image_args['y'] = 'y'
    image_args['dw'] = 'dw'
    image_args['dh'] = 'dh'
    image_args['source'] = self.source

    if self.color_mode_param == "Composite":
      self.image_renderer = self.fig.image_rgba(**image_args)
      self.color_bar = None

    elif self.color_mode_param == "Single":
      if not self.color_mapper:
        bounds = self.intensities_bounds[self._get_channel_index()]
        palette = self.palettes[str(self.colormap_param).lower()]
        self.color_mapper = bk.models.LinearColorMapper(low=bounds[0], high=bounds[1], palette=palette)

      self.image_renderer = self.fig.image(color_mapper=self.color_mapper, **image_args)

      # Add colorbar
      self.color_bar = bk.models.ColorBar(color_mapper=self.color_mapper, location=(0, 0))
      self.fig.add_layout(self.color_bar, 'right')

    else:
      raise ValueError(f"Invalid color mode: {self.color_mode_param}")

    # Add tolltips for the image.
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

    self.image_hover_tool = bk.models.tools.HoverTool(tooltips=image_tooltips,
                                                      renderers=[self.image_renderer])
    self.fig.add_tools(self.image_hover_tool)

    self._update_image_view()

    # Tell the drawer the figure is new
    if self.drawer:
      self.drawer.update_fig(self.fig, self.fig_pane)

  def _get_fig(self):
    return self.fig_pane

  def _update_fig(self):
    if self.fig_pane:
      self.fig_pane.param.trigger('object')

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

    channel_index = self._get_channel_index()

    if self.color_mode_param == "Composite":

      # Bound channels independently.
      images = self.image[self.time_param, :, self.z_param].copy()
      for i, bound in enumerate(self.intensities_bounds):
        images[i] = skimage.exposure.rescale_intensity(images[i], in_range=bound)

      # Create the composite image.
      colors = self.COMPOSITE_COLORS[:self.image_channel]
      frame = create_composite(images, colors)

      # Rescale to uint8 and add an alpha channel.
      frame = skimage.exposure.rescale_intensity(frame, out_range=(0, 255))
      frame = frame.astype('uint8')
      frame = np.insert(frame, 3, 255, axis=2)

    elif self.color_mode_param == "Single":
      frame = self.image[self.time_param, channel_index, self.z_param]

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

    # Update the drawer
    if self.drawer:
      self.drawer.cursor["time_index"] = self.time_param
      self.drawer.cursor["channel_index"] = channel_index
      self.drawer.cursor["z_index"] = self.z_param
      self.drawer.draw()

    self._plot_frame(frame, metadata)

  @param.depends('channel_param', watch=True)
  def _update_intensities_slider_bounds(self):
    """Update intensities slider bounds.
    """
    channel_index = self._get_channel_index()
    self.intensities_param = self.intensities_bounds[channel_index]
    self._update_intensities_range()
    self.param.trigger('intensities_param')

  @param.depends('intensities_param', watch=True)
  def _update_intensities_range(self):
    """Update intensities range.

    TODO: the widget does not seems to update when switching channels.
    """
    if self.color_mapper:
      channel_index = self._get_channel_index()
      self.intensities_bounds[channel_index] = tuple(self.intensities_param)
      self.color_mapper.low = self.intensities_bounds[channel_index][0]
      self.color_mapper.high = self.intensities_bounds[channel_index][1]

    if self.color_mode_param == "Composite":
      self._update_image_view()

  @param.depends('color_mode_param', watch=True)
  def _change_color_mode(self):
    """Update viewer to match the new color mode.
    """
    self._init_figure()
    self._plot_image()

  @param.depends('colormap_param', watch=True)
  def _update_colormap(self):
    """Update viewer to match the new colormap.
    """
    if self.color_mapper:
      self.color_mapper.palette = self.palettes[str(self.colormap_param).lower()]
      self._update_fig()

  @param.depends('time_param', 'channel_param', 'z_param', watch=True)
  def _get_image_info(self):
    """Get image informations about dimensions and current position
    in the viewer.

    Returns:
        infos: A Pandas dataframe converted to HTML code.
    """
    infos = []
    infos.append(('Image Type', self.image.dtype.name, ""))
    infos.append(('X', self.image_width, ""))
    infos.append(('Y', self.image_height, ""))
    if self.image_z > 1:
      infos.append(('Z', self.image_z, self.z_param))
    if self.image_time > 1:
      infos.append(('Time', self.image_time, self.time_param))
    if self.image_channel > 1:
      if self.channel_param:
        channel_index = self.channel_param
      else:
        channel_index = 'Channel 0'
      infos.append(('Channel', self.image_channel, self.channel_names.index(channel_index)))
    infos = pd.DataFrame(infos, columns=['Dimension', 'Size', 'Position'])
    return infos.to_html(index=False)

  def panel(self):
    """The image viewer as a panel to display.
    """

    self._set_css()

    # Widget to show image informations.
    info_widget = self._get_image_info
    info_container = pn.Column(pn.pane.Markdown('## Informations'), info_widget)

    # Widget with parameter widgets to control image viewer.
    parameters_widget = pn.Param(self.param,
                                 show_name=False,
                                 parameters=self.active_param_widgets,
                                 widgets=self.param_widgets,
                                 sizing_mode='scale_both')
    parameters_widget = pn.WidgetBox(parameters_widget)
    parameters_container = pn.Column(pn.pane.Markdown('## Viewer Configurations'), parameters_widget)

    # Assemble informations and parameters widgets.
    tools_widget = pn.Column(info_container, parameters_container)

    # Image container
    image_container = pn.Row(self._get_fig)

    # Log container
    log_container = self.log.panel(height=200)

    # Viewer layout
    main_pane_args = {}
    main_pane_args['css_classes'] = [f'viewer-{self.viewer_id}']
    main_pane_args['margin'] = 0

    if self.width and self.height:
      main_pane_args['width'] = self.width
      main_pane_args['height'] = self.height
    elif self.width:
      main_pane_args['width'] = self.width
      main_pane_args['sizing_mode'] = 'stretch_height'
    elif self.height:
      main_pane_args['height'] = self.height
      main_pane_args['sizing_mode'] = 'stretch_width'
    else:
      main_pane_args['sizing_mode'] = 'scale_both'

    gspec = pn.GridSpec(**main_pane_args)

    gspec[0, :1] = tools_widget
    gspec[0, 1:20] = image_container
    #gspec[1, :20] = viewer.drawer.panel()
    gspec[1, :20] = log_container

    return gspec
