import tifffile


def get_pixel_size(fname):
    """Return the pixel size from the ImageJ metadata."""
    tiff_obj = tifffile.TiffFile(str(fname))
    description = tiff_obj.pages.pages[0].description

    # First method
    scales = list(filter(lambda x: x.startswith('scales'), description.split("\n")))
    if len(scales) >= 1:
        scales = scales[0]
        pixel_size = scales.split('=')[1].split(',')[0]
        return float(pixel_size)

    # Second method
    raw_pixel_size, mul = tiff_obj.pages.pages[0].tags['YResolution'].value
    return 1 / (raw_pixel_size) * mul
