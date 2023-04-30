import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_result(nrow = None, ncol = None, res_stack = None):
    plt.figure(figsize=(12, 12))
    for i, (lbl, img) in enumerate(res_stack):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(lbl)
        plt.axis('off')
    plt.show()

image = cv2.imread('lena.jpg')
                        
thresh = 100
gray_ocv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh_img = gray_ocv.copy()

height, width = image.shape[:2]
for i in range(height):
    for j in range(width):
        if(thresh_img[i, j]) > thresh :
            thresh_img[i, j] = 255
        else: 
            thresh_img[i, j] = 0

show_result(1, 1, zip(['Manual Threshold'], [thresh_img]))


_, bin_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_BINARY)
_, binv_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_BINARY_INV)
_, mask_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_MASK)
_, otsu_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_OTSU)
_, to_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_TOZERO)
_, tinv_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_TOZERO_INV)
_, tri_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_TRIANGLE)
_, trunc_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_TRUNC)

thresh_labels = ['man', 'bin', 'binv', 'mask', 'otsu', 'to', 'tinv', 'tri', 'trunc']
thresh_img = [thresh_img, bin_thresh, binv_thresh, mask_thresh, otsu_thresh, to_thresh, tinv_thresh, tri_thresh, trunc_thresh]

show_result(3, 3, zip(thresh_labels, thresh_img))
