import numpy as np
import numpy.testing as npt
import anamic


def test_get_point_from_vector():
  vec = [5.44, 13.87]
  point = [10.01, 2.94]
  distance = 2.73

  point = anamic.geometry.get_point_from_vector(vec, point, distance)
  true_point = [11.00681351, 5.48150798]
  npt.assert_almost_equal(point, true_point)


def test_discretize_line():
  p1 = [0, 0]
  p2 = [1, 1]
  line = np.array([p1, p2])
  spacing = 0.3

  line = anamic.geometry.discretize_line(line, spacing)
  true_line = np.array([[0, 0],
                        [0.21213203, 0.21213203],
                        [0.42426407, 0.42426407],
                        [0.6363961, 0.6363961],
                        [1, 1]])

  npt.assert_almost_equal(line, true_line)


def test_get_normal_points():
  vec = [5.44, 13.87]
  point = [[10.01, 2.94], [55.12, 12.46]]
  distance = 5.4

  points = anamic.geometry.get_normal_points(vec, point, distance)

  true_points = np.array([4.98284135, 50.09284135, 4.91171904, 14.43171904,
                          15.03715865, 60.14715865, 0.96828096, 10.48828096])

  npt.assert_almost_equal(points.flatten(), true_points)


def test_get_rectangle_from_middle_line():
  p1 = [0, 0]
  p2 = [1, 1]
  rectangle_width = 2

  corners = anamic.geometry.get_rectangle_from_middle_line(p1, p2, rectangle_width)

  true_corners = np.array([-0.70710678, 0.70710678, 0.70710678, -0.70710678, 1.70710678,
                           0.29289322, 0.29289322, 1.70710678])

  npt.assert_almost_equal(corners.flatten(), true_corners)


def test_gget_mask_from_polygon():
  image = np.random.random((512, 512))
  polygon = [[10, 24], [65, 84], [100, 264], [5, 400], [347, 42]]

  mask1 = anamic.geometry.get_mask_from_polygon(image.shape, polygon, backend='matplotlib')

  assert image.shape == mask1.shape
  assert mask1.sum() == 41976

  mask2 = anamic.geometry.get_mask_from_polygon(image.shape, polygon, backend='skimage')

  assert image.shape == mask2.shape
  assert mask2.sum() == 41985
