import numpy as np
from scipy.fftpack import dct, idct

def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')

def qim_embed(c, b, delta=10):
    if b == 0:
        return delta * np.round(c / delta)
    else:
        return delta * (np.round(c / delta) + 0.5)

def qim_extract(c, delta=10):
    return 0 if (c/delta) % 1 < 0.25 else 1


def embed(image, watermark, delta=10):
    dct_img = dct2(image.astype(float))
    h, w = dct_img.shape

    k = 0
    for i in range(10, h-10):
        for j in range(10, w-10):
            if k < len(watermark):
                dct_img[i, j] = qim_embed(dct_img[i, j], watermark[k], delta)
                k += 1

    watermarked = idct2(dct_img)
    return np.clip(watermarked, 0, 255).astype(np.uint8)

def extract(image, size, delta=10):
    dct_img = dct2(image.astype(float))
    h, w = dct_img.shape

    extracted = []
    k = 0
    for i in range(10, h-10):
        for j in range(10, w-10):
            if k < size:
                extracted.append(qim_extract(dct_img[i, j], delta))
                k += 1

    return np.array(extracted)
