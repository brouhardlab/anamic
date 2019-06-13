import logging

import panel as pn
import param

from anamic.utils import css_dict_to_string


class LoggingWidget(param.Parameterized):
  widget = pn.pane.Markdown("", css_classes=[])

  def __init__(self, logger_name, enable=True, **kwargs):
    super().__init__(**kwargs)

    self.logger_name = logger_name
    self.enable = enable
    self.widget_id = id(self)
    self.log = None
    self._log_lines = []

    self._setup_logger()

  def info(self, message):
    self.log.info(message)

  def error(self, message):
    self.log.error(message)

  def warn(self, message):
    self.log.warning(message)

  def critical(self, message):
    self.log.critical(message)

  # pylint: disable=arguments-differ
  def debug(self, message):
    self.log.debug(message)

  def _setup_logger(self):
    self.log = logging.getLogger(self.logger_name)
    self.log.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
    custom_handler = logging.StreamHandler()
    custom_handler.setFormatter(formatter)
    custom_handler.emit = self._logging_handler
    self.log.addHandler(custom_handler)
    self.widget.css_classes.append(f'log-widget-{self.widget_id}')
    self._set_css()

  def _logging_handler(self, record):
    """Log a message in the UI.
    """
    if self.enable:
      message = self.log.handlers[0].format(record)
      self._log_lines.append(message)
      self.widget.object = "<br/>".join(self._log_lines[::-1])

  def _set_css(self):
    css = {}
    css[f'.log-widget-container-{self.widget_id}'] = {}
    css[f'.log-widget-container-{self.widget_id}']['background'] = '#fbfbfb'
    css[f'.log-widget-container-{self.widget_id}']['border'] = '1px #eaeaea solid !important'
    css[f'.log-widget-container-{self.widget_id}']['overflow-y'] = 'auto'
    css[f'.log-widget-container-{self.widget_id}']['border-radius'] = '0'
    css[f'.log-widget-container-{self.widget_id}']['margin'] = '10px !important'
    css[f'.log-widget-container-{self.widget_id}']['resize'] = 'both'
    css[f'.log-widget-container-{self.widget_id}']['min-width'] = '95% !important'
    css[f'.log-widget-{self.widget_id}'] = {}
    #css[f'.log-widget-{self.widget_id}']['min-width'] = '100%'

    css_string = css_dict_to_string(css)
    pn.extension(raw_css=[css_string])

  def _get_log_widget(self):
    if self.enable:
      return pn.Column(self.widget, height=150,
                       css_classes=[f'log-widget-container-{self.widget_id}'],
                       sizing_mode='stretch_width')
    return None

  def panel(self):
    return self._get_log_widget()
