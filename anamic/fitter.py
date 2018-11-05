import numpy as np
from scipy import ndimage

from . import geometry


def line_profile(image, point1, point2, pixel_size=1,
                 offset_start=0, offset_end=0,
                 spacing=0.1, line_thickness=0,
                 line_thickness_spacing=0.1,
                 normalized_intensities=True):
    """Get a line profile defined by an image and two points.
    
    Args:
        image: array.
        point1: array.
        point2: array.
        pixel_size: float, the size of a pixel in nm.
            Used to convert other arguments in pixel.
        offset_start: float, the distance in pixel beyond `point1`.
        offset_end: float, the distance in pixel beyond `point2`.
        spacing: float, when computing the points inside a line
            what distance should separate the points (pixel).
        line_thickness: float, the thickness of the line used. If > 0, the
            returned line profile will be the mean of the different lines used.
        line_thickness_spacing: float, same as `spacing` but in the  perpendicular
            direction when `spacing` is not 0.
        normalized_intensities: bool, normalize from 0 to 1 if True.
        
    Returns:
        x_profile: array, the x coordinates of the profile where unit is pixel.
        y_profile: array, the intensities values of the profile.
        line: array of 2x2, the start and end point location used.
    """
        
    offset_start_pixel = offset_start / pixel_size
    offset_end_pixel = offset_end / pixel_size

    # Calculate the vector of the line and its norm
    vec = point2 - point1

    # Get the coordinates of the points we'll use
    # to for line fitting.
    start_point = geometry.get_point_from_vector(-vec, point2, offset_start_pixel)
    end_point = geometry.get_point_from_vector(vec, point2, offset_end_pixel)
    line_tips = np.array([start_point, end_point])
    
    # Get points along the initial line.
    points = geometry.discretize_line(line_tips, spacing)
    
    if line_thickness > 0:
        
        # Get the two lines parallel to the initial line
        # at the distance defined by `line_thickness`.
        line_thickness_pixel = line_thickness / pixel_size
        normal_distance = line_thickness_pixel / 2
        vec = points[-1] - points[0]
        normal_points = geometry.get_normal_points(vec, points, normal_distance)
        
        lines = []
        # Iterate over the length of the initial line.
        for p1, p2 in np.rollaxis(normal_points, -1):
            line = np.array([p1, p2])
            thickness_points = geometry.discretize_line(line, line_thickness_spacing)
            lines.append(thickness_points)
        lines = np.array(lines).T
    else:
        lines = np.array([points]).T.swapaxes(1, 2)
        
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
    x_profile = np.arange(0, y_profile.shape[0]) * spacing
    
    return x_profile, y_profile, line_tips