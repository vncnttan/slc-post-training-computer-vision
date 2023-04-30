import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('lena.jpg')
gray_ocv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = image.shape[:2]

def manual_mean_filter(source, ksize):
    np_source = np.array(source)
    for i in range(height - ksize - 1):
        for j in range(width - ksize - 1):
            matrix = np.array(np_source[i: (i+ksize), j: (j+ksize)]).flatten()
            mean = np.mean(matrix)
            np_source[i+ksize//2, j+ksize//2] = mean
    return np_source


def manual_median_filter(source, ksize):
    np_source = np.array(source)
    for i in range(height - ksize - 1):
        for j in range(width - ksize - 1):
            matrix = np.array(np_source[i: (i+ksize), j: (j+ksize)]).flatten()
            median = np.median(matrix)
            np_source[i+ksize//2, j+ksize//2] = median
    return np_source

b, g, r = cv2.split(image)
mean_b = manual_mean_filter(b, 3)
mean_g = manual_mean_filter(g, 3)
mean_r = manual_mean_filter(r, 3)

median_b = manual_median_filter(b, 3)
median_g = manual_median_filter(g, 3)
median_r = manual_median_filter(r, 3)

merged_mean = cv2.merge((mean_b, mean_g, mean_r))
merged_median = cv2.merge((median_b, median_g, median_r))

blur_image = gray_ocv.copy()

blur = cv2.blur(blur_image, (3, 3))
median_blur = cv2.medianBlur(blur_image, 3)
gauss_blur = cv2.GaussianBlur(blur_image, (3, 3), 2.0)
bilateral_blur = cv2.bilateralFilter(blur_image, 3, 150, 150)

blur_labels = ['blur', 'median_blur', 'gauss blur', 'bilateral blur', 'merged mean', 'merged median']
blur_images = [blur, median_blur, gauss_blur, bilateral_blur, merged_mean, merged_median]

def show_result(nrow = None, ncol = None, res_stack = None):
    plt.figure(figsize=(12, 12))
    for i, (lbl, img) in enumerate(res_stack):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(lbl)
        plt.axis('off')
    plt.show()

show_result(2, 3, zip(blur_labels, blur_images))