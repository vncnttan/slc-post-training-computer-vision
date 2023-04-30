import cv2
import numpy as np
import matplotlib.pyplot as plt

image_obj = cv2.imread("marjan.png")
image_scn = cv2.imread("marjan_banyak.png")

SIFT = cv2.SIFT_create()
ORB = cv2.ORB_create()
AKAZE = cv2.AKAZE_create()

sift_kp_obj, sift_ds_obj = SIFT.detectAndCompute(image_obj, None)
sift_kp_scn, sift_ds_scn = SIFT.detectAndCompute(image_scn, None)

orb_kp_obj, orb_ds_obj = ORB.detectAndCompute(image_obj, None)
orb_kp_scn, orb_ds_scn = ORB.detectAndCompute(image_scn, None)

akaze_kp_obj, akaze_ds_obj = AKAZE.detectAndCompute(image_obj, None)
akaze_kp_scn, akaze_ds_scn = AKAZE.detectAndCompute(image_scn, None)

#SIFT, AKAZE - Euclidean ubah jadi float32 si descriptornya
sift_ds_obj = np.float32(sift_ds_obj)
sift_ds_scn = np.float32(sift_ds_scn)

akaze_ds_obj = np.float32(akaze_ds_obj)
akaze_ds_scn = np.float32(akaze_ds_scn)

#ORB - hamming
flann = cv2.FlannBasedMatcher(dict(algorithm = 1), dict(checks = 50))
bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

#matching
sift_match = flann.knnMatch(sift_ds_obj, sift_ds_scn, 2)
akaze_match = flann.knnMatch(akaze_ds_obj, akaze_ds_scn, 2)

orb_match = bfmatcher.match(orb_ds_obj, orb_ds_scn)
orb_match = sorted(orb_match, key = lambda x : x.distance)

#create masking
def createMasking(mask, match):
    for i, (fm, sm) in enumerate(match):
        if(fm.distance < 0.7 * sm.distance):
            mask[i] = [1, 0]
    return mask

sift_matches_mask = [[0, 0] for i in range(0, len(sift_match))]
akaze_matches_mask = [[0, 0] for i in range(0, len(akaze_match))]

sift_matches_mask = createMasking(sift_matches_mask, sift_match)
sift_matches_mask = createMasking(sift_matches_mask, sift_match)

sift_res = cv2.drawMatchesKnn(
    image_obj, sift_kp_obj,
    image_scn, sift_kp_scn,
    sift_match, None,
    matchColor=[255, 0, 0], singlePointColor=[0, 255, 0],
    matchesMask = sift_matches_mask
)

akaze_res = cv2.drawMatchesKnn(
    image_obj, akaze_kp_obj,
    image_scn, akaze_kp_scn,
    akaze_match, None,
    matchColor=[255, 0, 0], singlePointColor=[0, 255, 0],
    matchesMask = akaze_matches_mask
)

orb_res = cv2.drawMatches(
    image_obj, orb_kp_obj,
    image_scn, orb_kp_scn,
    orb_match[:20], None,
    matchColor=[255, 0, 0], singlePointColor=[0, 255, 0],
    flags=2
)

res_labels = ['sift', 'akaze', 'orb']
res_images = [sift_res, akaze_res, orb_res]
plt.figure(figsize=(12, 12))
for i, (lbl, img) in enumerate(zip(res_labels, res_images)):
    plt.subplot(2, 2, 2+i)
    plt.imshow(img, cmap='gray')
    plt.title(lbl)
plt.show()
