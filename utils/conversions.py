def convert_pixels_to_meters_util(pixel_dist, reference_height_meters, reference_height_pixels):
    return (pixel_dist * reference_height_meters) / reference_height_pixels


def convert_meters_to_pixels_util(meters_dist, reference_height_meters, reference_height_pixels):
    return (meters_dist * reference_height_pixels) / reference_height_meters
