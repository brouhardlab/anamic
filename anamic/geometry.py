import numpy as np
from skimage import draw
from matplotlib import path


def get_point_from_vector(vec, point, distance):
    """Given a vector get the coordinate of the point
    at a certain distance from the input point.

    Args:
      vec: array, vector.
      point: array, input point.
      distance: float, the distance.
    """
    vec = np.array(vec)
    point = np.array(point)
    norm = np.sqrt(np.sum(vec ** 2))
    return point + (vec / norm) * distance


def discretize_line(line, spacing):
    """Return a list points located at equidistance on the input line.

    The list will also include the input line points.

    Args:
      line: array, shape=2x2
      spacing: float, the distance between each points.

    """

    vec = line[1] - line[0]
    norm = np.sqrt(np.sum(vec ** 2))

    points = []

    distances = np.arange(0, np.round(norm), spacing)
    for d in distances:
        p = get_point_from_vector(vec, line[0], d)
        points.append(p)

    points.append(line[1])
    return np.array(points)


def get_normal_points(vec, points, distance):
    """From a vector and a list of points, get the point perpendicular
    to the vector at a specific distance from the input point.

    Args:
      vec: array, vector.
      points: array, input point (can be a single point
          or an array of points).
      distance: float.
    """
    vec = np.array(vec)
    points = np.array(points)

    norm = np.sqrt(np.sum(vec ** 2))

    # Get the normal vectors
    n1 = np.array([-vec[1], vec[0]])
    n2 = np.array([vec[1], -vec[0]])

    # Distance ratio
    t = distance / norm

    points1 = (1 - t) * points + t * (points + n1)
    points2 = (1 - t) * points + t * (points + n2)

    return np.array([points1.T, points2.T])


# pylint: disable=too-many-locals
def get_rectangle_from_middle_line(p1, p2, rectangle_width):
    """Get the rectangle corner points from two points defining the line crossing the
    rectangle in its middle.

    Args:
      p1: list or array, x and y of point 1.
      p2: list or array, x and y of point 2.
      rectangle_width: float, width of the rectangle.
    """

    p1 = np.array(p1)
    p2 = np.array(p2)

    norm = np.sqrt(np.sum((p1 - p2) ** 2))

    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    # Get normal vectors
    n1 = np.array([-dy, dx])
    n2 = np.array([dy, -dx])

    # Distance ratio
    t = (rectangle_width / 2) / norm

    # Get corner points
    corner1 = (1 - t) * p1 + t * (p1 + n1)
    corner2 = (1 - t) * p2 + t * (p2 + n2)
    corner3 = (1 - t) * p1 + t * (p1 + n2)
    corner4 = (1 - t) * p2 + t * (p2 + n1)
    corners = np.array([corner1, corner3, corner2, corner4])
    return corners


def get_mask_from_polygon(image_shape, polygon, backend="skimage"):
    """Get a mask image of pixels inside the polygon.

    Args:
      image_shape: tuple of size 2.
      polygon: Numpy array of dimension 2 (2xN).
      backend: str, matplotlib or skimage.
    """
    # pylint: disable=no-else-return
    if backend == "matplotlib":
        return get_mask_from_polygon_mpl(image_shape, polygon)
    elif backend == "skimage":
        return get_mask_from_polygon_skimage(image_shape, polygon)
    return None


def get_mask_from_polygon_mpl(image_shape, polygon):
    """Get a mask image of pixels inside the polygon.

    Args:
      image_shape: tuple of size 2.
      polygon: Numpy array of dimension 2 (2xN).
    """
    polygon = np.array(polygon)
    xx, yy = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    xx, yy = xx.flatten(), yy.flatten()
    indices = np.vstack((xx, yy)).T
    mask = path.Path(polygon).contains_points(indices)
    mask = mask.reshape(image_shape)
    return mask


def get_mask_from_polygon_skimage(image_shape, polygon):
    """Get a mask image of pixels inside the polygon.

    Args:
      image_shape: tuple of size 2.
      polygon: Numpy array of dimension 2 (2xN).
    """
    polygon = np.array(polygon)
    vertex_row_coords = polygon[:, 1]
    vertex_col_coords = polygon[:, 0]
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, image_shape
    )
    mask = np.zeros(image_shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask
