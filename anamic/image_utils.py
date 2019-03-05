import tifffile


def get_pixel_size(fname):
    """Return the pixel size from the ImageJ metadata."""
    tiff_obj = tifffile.TiffFile(str(fname))
    description = tiff_obj.pages.pages[0].description

    scales = list(filter(lambda x: x.startswith('scales'), description.split("\n")))[0]
    return scales.split('=')[1].split(',')[0]
