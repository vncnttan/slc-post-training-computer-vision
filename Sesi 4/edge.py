import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('fruits.jpg')
igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

laplace_8u = cv2.Laplacian(igray, cv2.CV_8U)
laplace_16s = cv2.Laplacian(igray, cv2.CV_16S)
laplace_32f = cv2.Laplacian(igray, cv2.CV_32F)
laplace_64f = cv2.Laplacian(igray, cv2.CV_64F)

def show_result(nrow = None, ncol = None, res_stack = None):
    plt.figure(figsize=(12, 12))
    for i, (lbl, img) in enumerate(res_stack):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(lbl)
        plt.axis('off')
    plt.show()

laplace_labels = ['8u', '16s', '32f', '64f']
laplace_images = [laplace_8u, laplace_16s, laplace_32f, laplace_64f]

show_result(2, 2, zip(laplace_labels, laplace_images))

ksize = 3
sobel_x = cv2.Sobel(igray, cv2.CV_32F, 1, 0, ksize)
sobel_y = cv2.Sobel(igray, cv2.CV_32F, 0, 1, ksize)

sobel_labels = ['sobel x', 'sobel y']
sobel_images = [sobel_x, sobel_y]

show_result(1, 2, zip(sobel_labels, sobel_images))

merged_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
merged_sobel *= 255/merged_sobel.max()

show_result(1, 1, zip(['merged sobel'], [merged_sobel]))


canny_50100 = cv2.Canny(igray, 50, 100)
canny_50150 = cv2.Canny(igray, 50, 150)
canny_75150 = cv2.Canny(igray, 75, 150)
canny_75225 = cv2.Canny(igray, 75, 225)

canny_labels = ["canny_50100", "canny_50150", "canny_75150", "canny_75225"]
canny_images = [canny_50100, canny_50150, canny_75150, canny_75225]

show_result(2, 2, zip(canny_labels, canny_images))