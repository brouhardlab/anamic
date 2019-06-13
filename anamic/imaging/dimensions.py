import numpy as np


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
