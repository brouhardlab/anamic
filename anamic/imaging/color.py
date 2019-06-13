import numpy as np
import skimage
import matplotlib
import colorcet as cc
import bokeh as bk


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
    colored_images.append(im)
  colored_images = np.array(colored_images)
  composite = np.sum(colored_images, axis=0)
  return composite


def get_palettes():
  palettes = {}
  palettes['grey'] = bk.palettes.Greys256
  palettes['inferno'] = bk.palettes.Inferno256
  palettes['magma'] = bk.palettes.Magma256
  palettes['plasma'] = bk.palettes.Plasma256
  palettes['viridis'] = bk.palettes.Viridis256
  palettes['cividis'] = bk.palettes.Cividis256
  palettes['fire'] = cc.fire
  palettes['rainbow'] = cc.rainbow
  palettes['red'] = cc.kr
  palettes['green'] = cc.kg
  palettes['blue'] = cc.kb
  return palettes
