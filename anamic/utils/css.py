def css_dict_to_string(css_dict):
  return CSS(css_dict).to_string()

class CSS:
  """A simple class to convert a dict to a CSS string.
  """

  def __init__(self, obj):
    self.obj = obj
    self.__data = {}
    self.__parse(obj)

  def to_string(self):
    return self.__build()

  def __repr__(self):
    return self.__build()

  def __build(self, string=''):
    for key, value in sorted(self.__data.items()):
      if self.__data[key]:
        string += key[1:] + ' {\n' + ''.join(value) + '}\n\n'
    return string

  def __parse(self, obj, selector=''):
    for key, value in obj.items():
      if hasattr(value, 'items'):
        rule = selector + ' ' + key
        self.__data[rule] = []
        self.__parse(value, rule)
      else:
        prop = self.__data[selector]
        prop.append('\t%s: %s;\n' % (key, value))
