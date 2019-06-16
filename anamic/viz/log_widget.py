import logging

import panel as pn

from anamic.utils import css_dict_to_string


class LoggingWidget():

  def __init__(self, logger_name, enable=True):
    super().__init__()

    self.logger_name = logger_name
    self.enable = enable
    self.widget_id = id(self)
    self.log = None
    self._log_lines = []
    self.widget = pn.pane.Markdown("", css_classes=[])

    self._setup_logger()

  def info(self, message):
    self.log.info(str(message))

  def error(self, message):
    self.log.error(str(message))

  def warning(self, message):
    self.log.warning(str(message))

  def critical(self, message):
    self.log.critical(str(message))

  # pylint: disable=arguments-differ
  def debug(self, message):
    self.log.debug(str(message))

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
      self._log_lines.append(str(message))
      self.widget.object = "<br/>".join(self._log_lines[::-1])

  def _set_css(self):
    css = {}
    css[f'.log-widget-container-{self.widget_id}'] = {}
    css[f'.log-widget-container-{self.widget_id}']['width'] = '98% !important'
    css[f'.log-widget-container-{self.widget_id}']['background'] = '#f4f4f4'
    css[f'.log-widget-container-{self.widget_id}']['border'] = '1px #eaeaea solid !important'
    css[f'.log-widget-container-{self.widget_id}']['overflow-y'] = 'scroll'
    css[f'.log-widget-container-{self.widget_id}']['overflow-x'] = 'hidden'
    css[f'.log-widget-container-{self.widget_id}']['border-radius'] = '0'
    css[f'.log-widget-{self.widget_id}'] = {}
    css[f'.log-widget-{self.widget_id}']['width'] = '100% !important'

    css_string = css_dict_to_string(css)
    pn.extension(raw_css=[css_string])

  def _get_log_widget(self, height=250):
    if self.enable:
      return pn.Column(pn.pane.Markdown('## Logging'),
                       self.widget,
                       min_height=height,
                       max_height=height,
                       sizing_mode='stretch_both',
                       css_classes=[f'log-widget-container-{self.widget_id}'])
    return None

  def panel(self, height=250):
    return self._get_log_widget(height=height)
