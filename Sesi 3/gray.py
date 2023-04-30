import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('lena.jpg')
def show_result(nrow = None, ncol = None, res_stack = None):
    plt.figure(figsize=(12, 12))
    for i, (lbl, img) in enumerate(res_stack):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(lbl)
        plt.axis('off')
    plt.show()

gray_ocv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_avg = np.dot(image, [0.33, 0.33, 0.33])

image_b, image_g, image_r = image[:, :, 0], image[:, :, 1], image[:, :, 2]

max_cha = max(np.max(image_b), np.max(image_g), np.max(image_r))
min_cha = min(np.max(image_b), np.max(image_g), np.max(image_r))

gray_light = np.dot(image, [(max_cha + min_cha) / 2, (max_cha + min_cha) / 2, (max_cha + min_cha) / 2])
gray_lum = np.dot(image, [0.07, 0.71, 0.21])
gray_wag = np.dot(image, [0.114, 0.587, 0.299])

gray_labels = ['ocv', 'avg', 'lig', 'lum', 'wag']
gray_images = [gray_ocv, gray_avg, gray_light, gray_lum, gray_wag]

show_result(3, 2, zip(gray_labels, gray_images))