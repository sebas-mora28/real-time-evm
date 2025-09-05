import cv2
import numpy as np


def get_image_gaussian_pyramid(image, levels):
    image_shape = [image.shape[:2]]
    downsampled_image = image.copy()

    for _ in range(levels):
        downsampled_image = cv2.pyrDown(src=downsampled_image)
        image_shape.append(downsampled_image.shape[:2])

    gaussian_pyramid = downsampled_image
    for current_level in range(levels):
       gaussian_pyramid = cv2.pyrUp(src=gaussian_pyramid, dst=image_shape[levels - current_level - 1])

    return gaussian_pyramid


def generate_gaussian_pyramids(images, levels):
    gaussian_pyramids = np.zeros_like(images, dtype=np.float32)
    print(gaussian_pyramids.shape)
    for i in range(len(images)):
        
        gaussian_pyramids[i] = get_image_gaussian_pyramid(images[i], levels); 

    return gaussian_pyramids