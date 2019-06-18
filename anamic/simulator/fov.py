from tqdm.auto import tqdm
import numpy as np
from scipy import ndimage
from skimage import morphology

from . import dimers_builder
from . import MicrotubuleSimulator


def pick_value(values, prob=None):
  # pylint: disable=no-else-return
  if isinstance(prob, list):
    return np.random.choice(values, p=prob)
  elif prob == 'poisson':
    return np.random.poisson(**values)
  elif prob == 'normal':
    return np.random.normal(**values)
  elif prob == 'uniform':
    return np.random.choice(values)
  elif prob is None:
    return values
  else:
    raise Exception(f"{values} and {prob} are not valid.")


def sample_parameters(n_microtubules_to_sample, parameters, floating_parameters):
  # Here we generate a list of parameters
  # to generate microtubules
  parameters_list = []
  for _ in tqdm(range(n_microtubules_to_sample), total=n_microtubules_to_sample):
    args = {}
    args.update(parameters.copy())

    for k, v in floating_parameters.items():
      value = pick_value(**v)
      if k == 'taper_length_nm':
        value = max(10, value)
      args[k] = value

    parameters_list.append(args)
  return parameters_list


# pylint: disable=too-many-locals,too-many-statements
def create_fov(image_size_pixel, pixel_size, microtubule_parameters, image_parameters, return_positions=False):
  """
  Args:
      parameters_list:
      image_parameters:
      mask_rectangle_width:
  """
  image_size_nm = int(image_size_pixel * pixel_size)

  # Choose parameters
  n_mt = pick_value(**image_parameters['n_mt'])
  signal_mean = pick_value(**image_parameters['signal_mean'])
  # signal_std = pick_value(**image_parameters['signal_std'])
  bg_mean = pick_value(**image_parameters['bg_mean'])
  bg_std = pick_value(**image_parameters['bg_std'])
  noise_factor = pick_value(**image_parameters['noise_factor'])

  params_list = []
  for _ in range(n_mt):
    params = {}
    for k, v in microtubule_parameters.items():
      if isinstance(v, dict):
        params[k] = pick_value(**v)
      else:
        params[k] = v
    params_list.append(params)

  # Create the microtubules (2d positions)
  mts = []
  for params in tqdm(params_list, total=len(params_list), leave=True):

    # Taper length cannot be bigger than half the length of a microtubule.
    params['taper_length_nm'] = min(0.5 * params['mt_length_nm'], params['taper_length_nm'])

    dimers = dimers_builder(params['n_pf'], params['mt_length_nm'], params['taper_length_nm'])
    ms = MicrotubuleSimulator(dimers)
    ms.parameters.update(params)
    ms.build_positions(apply_random_z_rotation=True, show_progress=False)
    ms.label()
    ms.project()
    ms.random_rotation_projected()
    mts.append(ms)

  # Locate each microtubules on a 2D grid (translation)
  x_centers = np.random.randint(0, image_size_nm + 1, n_mt)
  y_centers = np.random.randint(0, image_size_nm + 1, n_mt)
  centers = np.array([x_centers, y_centers]).T
  for ms, center in zip(mts, centers):
    ms.positions[['x_proj_rotated', 'y_proj_rotated']] += center
    ms.positions[['x_pixel', 'y_pixel']] = ms.positions[['x_proj_rotated', 'y_proj_rotated']] / pixel_size

  # Discretize all the dimer's positions
  x = []
  y = []
  for ms in mts:
    selected_dimers = ms.positions[(ms.positions['visible']) & (ms.positions['labeled'])]
    x += selected_dimers['x_proj_rotated'].tolist()
    y += selected_dimers['y_proj_rotated'].tolist()
  dimers = np.array([x, y])

  x_bins = np.arange(0, image_size_nm + 1, pixel_size)
  y_bins = np.arange(0, image_size_nm + 1, pixel_size)
  discrete_image, _, _ = np.histogram2d(dimers[0], dimers[1], bins=[x_bins, y_bins])

  # Convolve and generate the FOV
  # All mt objects are supposed to have the same PSF
  ms.generate_psf()
  psf = ms.psf

  # Convolve the image with the PSF
  image = ndimage.convolve(discrete_image, psf, mode="constant")

  # Scale image to 0 - 1
  image /= image.max()

  # Bring signal to signal_mean and background to bg_mean
  image = image * (signal_mean - bg_mean) + bg_mean

  # Create read noise
  read_noise = np.random.normal(loc=0, scale=bg_std, size=image.shape) * noise_factor
  image += read_noise

  # Generate the masks
  masks = []
  for ms in mts:
    #selected_dimers = ms.positions[(ms.positions['visible'] == True) & (ms.positions['labeled'] == True)]
    selected_dimers = ms.positions
    x = selected_dimers['x_proj_rotated']
    y = selected_dimers['y_proj_rotated']
    mask, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    mask = mask > 0
    mask = mask.astype('uint8')
    # Dilate mask
    mask = morphology.dilation(mask, morphology.square(3))
    masks.append(mask)
  masks = np.array(masks)

  if return_positions:
    return image, masks, mts

  return image, masks
