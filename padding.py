import numpy as np

def mirror_padding(image, PADDING_SIZE):
    image_pixels = np.array(image)
    padding_image = np.pad(image_pixels, ((PADDING_SIZE, PADDING_SIZE), (PADDING_SIZE, PADDING_SIZE)), "edge")
    return padding_image