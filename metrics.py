import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def compute_psnr(original, watermarked):
    return peak_signal_noise_ratio(original, watermarked)

def compute_ber(original, extracted):
    return np.sum(original != extracted) / len(original)
