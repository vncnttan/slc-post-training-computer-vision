import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('model.jpg')
igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def showResult(label = None, image = None, cmap = None ):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.hist(image.flat, bins=256, range=(0, 256)) # bins adalah pembagian dari histogramnya
    plt.title(label)
    plt.xlabel('Intensity value')
    plt.ylabel('Intensity quantity')
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

normal_image = igray.copy()
# showResult("Normal Image", normal_image, 'gray')

# Equalization using equalization histogram
nequ_hist = cv2.equalizeHist(igray)
# showResult("Nequ", nequ_hist, 'gray')

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cequ_hist = clahe.apply(igray)
# showResult("Cequ", cequ_hist, 'gray')


hist_labels = ['normal', 'nequ', 'cequ']
hist_images = [normal_image, nequ_hist, cequ_hist]

plt.figure(figsize=(12, 12))
for i, (lbl, img) in enumerate(zip(hist_labels, hist_images)):
    plt.subplot(3, 1, i+1)
    plt.hist(img.flat, bins=256, range=(0, 256)) # bins adalah pembagian dari histogramnya
    plt.title(lbl)
    plt.xlabel('Intensity value')
    plt.ylabel('Intensity quantity')
plt.show

plt.figure(figsize=(12, 12))
for i, (lbl, img) in enumerate(zip(hist_labels, hist_images)):
    plt.subplot(1, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(lbl)
    plt.axis('off')
plt.show()
