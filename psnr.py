import numpy as np
import math

MAX_VALUE = 255

def calc_psnr(original_image, filtered_image):
    original_pixels = np.array(original_image, dtype = "float64")
    filtered_pixels = np.array(filtered_image, dtype = "float64")
    if(original_pixels.size != filtered_pixels.size):
        print("Input image size is different")
        exit(8)
    square = (original_pixels - filtered_pixels) ** 2 #画素の2乗誤差
    MSE = np.sum(square) / original_pixels.size
    PSNR = 10 * math.log10(MAX_VALUE ** 2 / MSE)
    return PSNR