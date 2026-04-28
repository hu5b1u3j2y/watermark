import numpy as np
import cv2

def gaussian_noise(image, sigma=10):
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def jpeg_compression(image, quality=50):
    cv2.imwrite("temp.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imread("temp.jpg", 0)
