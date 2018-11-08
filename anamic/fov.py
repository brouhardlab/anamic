import tqdm
import numpy as np
from scipy import ndimage

from . import simulator


def pick_value(values, prob):
    if isinstance(prob, list):
        return np.random.choice(values, p=prob)
    elif prob == 'poisson':
        return np.random.poisson(**values)
    elif prob == 'normal':
        return np.random.normal(**values)
    elif prob == 'uniform':
        return np.random.choice(values)
    
    
def sample_parameters(n_microtubules_to_sample, parameters, floating_parameters):
    # Here we generate a list of parameters
    # to generate microtubules
    parameters_list = []
    for i in tqdm.tqdm_notebook(range(n_microtubules_to_sample), total=n_microtubules_to_sample):
        args = {}
        args.update(parameters.copy())

        for k, v in floating_parameters.items():
            value = pick_value(**v)
            if k == 'taper_length_nm':
                value = max(10, value)
            args[k] = value

        parameters_list.append(args)
    return parameters_list


def create_fov(image_size_pixel, pixel_size, parameters_list, image_parameters):
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
    signal_std = pick_value(**image_parameters['signal_std'])
    bg_mean = pick_value(**image_parameters['bg_mean'])
    bg_std = pick_value(**image_parameters['bg_std'])
    noise_factor = pick_value(**image_parameters['noise_factor'])
    
    # Pick from `parameters_list` a bunch of microtubule parameters
    params_list = np.random.choice(parameters_list, n_mt)
    
    # Create the microtubules (2d positions)
    mts = []
    for params in tqdm.tqdm_notebook(params_list, total=len(params_list), leave=False):
        dimers = simulator.dimers_builder(params['n_pf'], params['mt_length_nm'], params['taper_length_nm'])
        ms = simulator.MicrotubuleSimulator(dimers)
        ms.parameters.update(params)
        ms.build_positions(apply_random_z_rotation=True, show_progress=False)
        ms.label()
        ms.project()
        ms.random_rotation_projected()
        mts.append(ms)
    
    # Locate each microtubules on a 2D grid (translation)
    x_centers = np.random.randint(0, image_size_nm, n_mt)
    y_centers = np.random.randint(0, image_size_nm, n_mt)
    centers = np.array([x_centers, y_centers]).T
    for ms, center in zip(mts, centers):
        ms.positions[['x_proj_rotated', 'y_proj_rotated']] += center
        ms.positions[['x_pixel', 'y_pixel']] = ms.positions[['x_proj_rotated', 'y_proj_rotated']] / pixel_size
        
    # Discretize all the dimer's positions
    x = []
    y = []
    for ms in mts:
        selected_dimers = ms.positions[(ms.positions['visible'] == True) & (ms.positions['labeled'] == True)]
        x += selected_dimers['x_proj_rotated'].tolist()
        y += selected_dimers['y_proj_rotated'].tolist()
    dimers = np.array([x, y])
    
    x_bins = np.arange(0, image_size_nm, pixel_size)
    y_bins = np.arange(0, image_size_nm, pixel_size)
    discrete_image, _, _ = np.histogram2d(dimers[0], dimers[1], bins=[x_bins, y_bins])
    
    # Convolve and generate the FOV
    # All mt objects are supposed to have the same PSF
    ms._generate_psf()
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
        masks.append(mask)
    masks = np.array(masks)
    
    return image, masks