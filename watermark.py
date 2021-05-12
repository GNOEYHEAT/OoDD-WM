import math
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import color
from skimage.util import random_noise 


SEED=1011


def scaleSpectrum(A):
    return np.real(np.log10(np.absolute(A) + np.ones(A.shape)))

def randomVector(seed, length):
    np.random.seed(seed)
    return [np.random.choice([0,1]) for _ in range(length)]

def applyWatermark(imageMatrix, watermarkMatrix, alpha):
    shiftedDFT = fftshift(fft2(imageMatrix))
    watermarkedDFT = shiftedDFT + alpha * watermarkMatrix
    watermarkedImage = ifft2(ifftshift(watermarkedDFT))
    return watermarkedImage

def makeWatermark(imageShape, radius, secretKey, vectorLength=50):

    watermark = np.zeros(imageShape)
    center = (int(imageShape[0] / 2) + 1, int(imageShape[1] / 2) + 1)
    vector = randomVector(secretKey, vectorLength)

    x = lambda t: center[0] + int(radius * math.cos(t * 2 * math.pi / vectorLength))
    y = lambda t: center[1] + int(radius * math.sin(t * 2 * math.pi / vectorLength))
    indices = [(x(t), y(t)) for t in range(vectorLength)]

    for i,location in enumerate(indices):
        watermark[location] = vector[i]

    return watermark

def decodeWatermark(image, secretKey):
    pass

def watermarking(images, alpha):
    
    temp_images=[]
    for image in images:
        lab = color.rgb2lab(image)
        gray = color.rgb2gray(image)
        watermark = makeWatermark(gray.shape, min(gray.shape) / 4, secretKey=SEED)
        watermarked = np.real(applyWatermark(lab[:,:,0], watermark, alpha))
        lab[:,:,0] = watermarked
        rgb = color.lab2rgb(lab)
        temp_images.append(rgb)

    watermarked_images=np.stack(temp_images)
    
    return watermarked_images