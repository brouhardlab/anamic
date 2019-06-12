import unittest

import numpy as np
import panel as pn
import anamic


class TestImageViewer(unittest.TestCase):

  def test_dim_below_2(self):
    image = np.random.random((15,))
    with self.assertRaises(ValueError):
      anamic.viz.ImageViewer(image=image, dimension_order='TCXY')

  def test_dim_above_5(self):
    image = np.random.random((15, 15, 15, 15, 15, 15))
    with self.assertRaises(ValueError):
      anamic.viz.ImageViewer(image=image, dimension_order='TCXYSA')

  def test_mismatch_dimensions(self):
    image = np.random.random((15, 15, 15, 15, 15))
    with self.assertRaises(ValueError):
      anamic.viz.ImageViewer(image=image, dimension_order='TXY')


def test_image_viewer_init():
  image = np.random.random((15, 15, 15, 15, 15))
  viewer = anamic.viz.ImageViewer(image=image, dimension_order='TCZXY')
  assert isinstance(viewer.panel(), pn.layout.ListPanel)
  # TODO: maybe it's possible to more checks about the UI.
