import tifffile


def get_pixel_size(fname):
  """Return the pixel size in um."""
  tiff_obj = tifffile.TiffFile(str(fname))
  description = tiff_obj.pages.pages[0].description

  # From LSM metadata
  if tiff_obj.is_lsm:
    #pylint: disable=unsubscriptable-object
    pixel_size_m = tiff_obj.lsm_metadata['VoxelSizeX']
    return pixel_size_m / 1e-6

  # From ImageJ description
  scales = list(filter(lambda x: x.startswith('scales'), description.split("\n")))
  if len(scales) >= 1:
    scales = scales[0]
    pixel_size = scales.split('=')[1].split(',')[0]
    return float(pixel_size)

  # From ImageJ tags
  if 'YResolution' in tiff_obj.pages.pages[0].tags.keys():
    raw_pixel_size, mul = tiff_obj.pages.pages[0].tags['YResolution'].value
    return 1 / (raw_pixel_size) * mul

  raise Exception("Cant't find the pixel size from the metadata.")
