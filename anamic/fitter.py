import numpy as np
from scipy import ndimage
import lmfit
import scipy

from . import geometry


def get_thick_line(point1, point2, length_spacing=0.1, line_thickness=0, width_spacing=0.1):
    """Construct a list of points representing a discrete line. If `line_thickness` > 0 then
    the line is composed of multiple parallel line to each other with a width equal to
    `line_thickness`.
    
    Args:
        point1: 1D array or float.
        point2: 1D array or float.
        length_spacing: float, when computing the points inside a line
            what distance should separate the points (pixel).
        line_thickness: float, the thickness of the line used centered on
            the input line (pixel).
        width_spacing: float, same as `length_spacing` but in the  perpendicular
            direction when `line_thickness` is not 0.
        
    Returns:
        points: 3D array of shape [2xWxL] where W are points along the thickness of the line and L along its length.
    """
    
    line_tips = np.array([point1, point2])
    
    # Get points along the initial line.
    points = geometry.discretize_line(line_tips, length_spacing)
    
    if line_thickness > 0:
        
        # Get the two lines parallel to the initial line
        # at the distance defined by `line_thickness`.
        normal_distance = line_thickness / 2
        vec = points[-1] - points[0]
        normal_points = geometry.get_normal_points(vec, points, normal_distance)
        
        lines = []
        # Iterate over the length of the initial line.
        for p1, p2 in np.rollaxis(normal_points, -1):
            line = np.array([p1, p2])
            thickness_points = geometry.discretize_line(line, width_spacing)
            lines.append(thickness_points)
        lines = np.array(lines).T
    else:
        lines = np.array([points]).T.swapaxes(1, 2)
        
    return lines


def line_profile(image, point1, point2,
                 length_spacing=0.1,
                 line_thickness=0,
                 width_spacing=0.1,
                 normalized_intensities=True):
    """Get a line profile defined by an image and two points.
    
    Args:
        image: 2D array.
        point1: 1D array or float.
        point2: 1D array or float.
        length_spacing: float, when computing the points inside a line
            what distance should separate the points (pixel).
        line_thickness: float, the thickness of the line used centered on
            the input line (pixel).
        width_spacing: float, same as `length_spacing` but in the  perpendicular
            direction when `line_thickness` is not 0.
        normalized_intensities: bool, normalize from 0 to 1 if True.
        
    Returns:
        x_profile: array, the x coordinates of the profile where unit is pixel.
        y_profile: array, the intensities values of the profile.
    """
     
    lines = get_thick_line(point1, point2, length_spacing=length_spacing, 
                           line_thickness=line_thickness, width_spacing=width_spacing)
        
    # Get the intensity profile of the lines.
    y_profiles = ndimage.map_coordinates(image, lines.reshape(2, -1), order=3, mode='constant')
    y_profiles = y_profiles.reshape(lines.shape[1:])

    # Get the mean profile
    y_profile = y_profiles.mean(axis=0)

    if normalized_intensities:
        # Normalize the profile between 0 and 1.
        # FUTURE: could be modified.
        y_profile = y_profile / y_profile.max()

    # Define the x values of the profile so the unit is a pixel.
    x_profile = np.arange(0, y_profile.shape[0]) * length_spacing
    
    return x_profile, y_profile


def perpendicular_line_fit(lines, image, length_spacing, fit_threshold):
    """From a list of lines, fit each line to a Gaussian and record the `mu` value for each fit and compute the position
    in the image of this value.
    
    Args:
        lines: see what returns `get_thick_line()`.
        image: 2D array.
        length_spacing: float, `get_thick_line()`.
        fit_threshold: float, fit with a `mu` stderr above this value will be discarded.
    """
    
    def gaussian_wall(x, mu, sigma, mt, bg):
        return mt * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bg

    model= lmfit.Model(gaussian_wall)

    # This threshold is sensitive to `length_spacing`
    # TODO: I am not sure this is the best condition
    # to filter out bad fits.
    mu_stderr_threshold = fit_threshold

    args = {}
    args['length_spacing'] = length_spacing  # pixel
    args['line_thickness'] = 0
    args['normalized_intensities'] = True

    fitted_line = []
    best_fit = None
    for line in np.rollaxis(lines, -1):
        point1, point2 = line[:, 0], line[:, -1]
        x_profile, y_profile = line_profile(image, point1, point2, **args)

        fit_params = {}
        fit_params['mu'] = lmfit.Parameter('mu', value=x_profile[-1] / 2, min=0, max=x_profile[-1])
        fit_params['sigma'] = lmfit.Parameter('sigma', value=100, vary=True, min=0, max=x_profile[-1])
        fit_params['mt'] = lmfit.Parameter('mt', value=50, vary=True, min=0)
        fit_params['bg'] = lmfit.Parameter('bg', value=50, vary=True, min=0)
        fit_result = model.fit(y_profile, x=x_profile, **fit_params)

        # We discard fit with a bigger stderr than a threshold.
        if fit_result.params['mu'].stderr and fit_result.params['mu'].stderr < mu_stderr_threshold:
            # Distance between point 1 and the fitted center
            d = fit_result.best_values['mu']
            vec = point2 - point1

            # Get the point at a certain distance d from point1
            line_center = geometry.get_point_from_vector(vec, point1, d)
            fitted_line.append(line_center)

    fitted_line = np.array(fitted_line)
    return fitted_line


def tip_line_fit(point1, point2, image, length_spacing, line_thickness, width_spacing):
    """Fit the tip of a line to a complementary error function, 1 - erf(x).
    
    Args:
        point1: 1D array or float.
        point2: 1D array or float.
        image: 2D array.
        length_spacing: float, `get_thick_line()`.
        line_thickness: float, `get_thick_line()`.
        width_spacing: float, `get_thick_line()`.
    """
    
    profile_parameters = {}
    profile_parameters['length_spacing'] = length_spacing  # pixel
    profile_parameters['line_thickness'] = line_thickness # pixel
    profile_parameters['width_spacing'] = width_spacing  # pixel
    profile_parameters['normalized_intensities'] = True

    x_profile, y_profile = line_profile(image, point1, point2, **profile_parameters)

    def errorfunction(x, mu, sigma, mt, bg):
        return bg + (0.5 * mt * scipy.special.erfc((x - mu) / (np.sqrt(2) * sigma)))    

    model = lmfit.Model(errorfunction)

    fit_params = {}
    fit_params['mu'] = lmfit.Parameter('mu', value=x_profile[-1] / 2, min=0, max=x_profile[-1])
    fit_params['sigma'] = lmfit.Parameter('sigma', value=1, vary=True, min=0, max=x_profile[-1])
    fit_params['mt'] = lmfit.Parameter('mt', value=1, vary=True, min=0)
    fit_params['bg'] = lmfit.Parameter('bg', value=1, vary=True, min=0)
    fit_result = model.fit(y_profile.copy(), x=x_profile.copy(), **fit_params)
    
    return x_profile, y_profile, fit_result, errorfunction


def microtubule_tip_fitter(tip_start, tip_end, image, get_thick_line_args, perpendicular_line_fit_args,
                           offset_start, offset_end, tip_fit_args):
    """
    Args:
        tip_start:
        tip_end:
        image: 
        get_thick_line_args:
        perpendicular_line_fit_args:
        offset_start:
        offset_end:
        tip_fit_args:
    """
    lines = get_thick_line(tip_start, tip_end, **get_thick_line_args)
    
    fitted_line = perpendicular_line_fit(lines, image, **perpendicular_line_fit_args)

    # Now we fit the best line from those points
    a, b = np.polyfit(fitted_line[:, 1], fitted_line[:, 0], deg=1)
    new_point1 = np.array([a * fitted_line[0, 1] + b, fitted_line[0, 1]])
    new_point2 = np.array([a * fitted_line[-1, 1] + b, fitted_line[-1, 1]])

    # This the new line defining the microtubule
    new_line = np.array([new_point1, new_point2])
    
    # Now we fit the microtubule using a line profile with a defined thickness.

    # We define a new line we are going to use for the fitting process
    line = np.array([tip_start, tip_end])
    # Calculate the vector of the line and its norm
    vec = tip_end - tip_start
    # Get the coordinates of the points we'll use
    # to for line fitting.
    start_point = geometry.get_point_from_vector(-vec, tip_end, offset_start)
    end_point = geometry.get_point_from_vector(vec, tip_end, offset_end)
    line_fit_tips = np.array([start_point, end_point])
    
    # Fit the tip
    return [line_fit_tips] + list(tip_line_fit(line_fit_tips[0], line_fit_tips[1], image, **tip_fit_args))
