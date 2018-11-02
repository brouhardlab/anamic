import numpy as np


def compute_snr(signal_mean, signal_std, bg_mean, bg_std):
    """Compute the SNR of a given object in an image.
        
    Args:
        signal_mean: float, mean signal value.
        signal_std: float, std signal value.
        bg_mean: float, mean background value.
        bg_std: float, std background value.
    """
    return (signal_mean - bg_mean) / np.sqrt(signal_std**2 + bg_std**2)


def get_rectangle_from_middle_line(p1, p2, rectangle_width):
    """Get the rectangle corner points from two points defining the line crossing the
    rectangle in its middle.
    
    Args:
        p1: list or array, x and y of point 1.
        p2: list or array, x and y of point 2.
        rectangle_width: float, width of the rectangle.
    """
    d = np.sqrt(np.sum((p1 - p2) ** 2))

    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1

    # Get normal vectors
    n1 = np.array([-dy, dx])
    n2 = np.array([dy, -dx])

    # Distance ratio
    t = (rectangle_width / 2) / d

    # Get corner points
    corner1 = (1 - t) * p1 + t * (p1 + n1)
    corner2 = (1 - t) * p2 + t * (p2 + n2)
    corner3 = (1 - t) * p1 + t * (p1 + n2)
    corner4 = (1 - t) * p2 + t * (p2 + n1)
    corners = np.array([corner1, corner3, corner2, corner4])
    return corners


def get_mask_from_polygon(image, polygon):
    """Get a mask image of pixels inside the polygon.
    
    Args:
        image: Numpy array of dimension 2.
        polygon: Numpy array of dimension 2 (2xN).
    """
    from matplotlib import path
    xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xx, yy = xx.flatten(), yy.flatten()
    indices = np.vstack((xx, yy)).T
    mask = path.Path(polygon).contains_points(indices)
    mask = mask.reshape(image.shape)
    return mask