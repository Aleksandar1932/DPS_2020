import cv2
import numpy as np
from matplotlib import pyplot as plt

# noinspection SpellCheckingInspection
FLANN_INDEX_KDTREE = 0


def find_keypoints(img_src):
	grey = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
	descriptor = cv2.xfeatures2d.SIFT_create()
	kp, ds = descriptor.detectAndCompute(grey, None)
	# cv2.drawKeypoints(image=img_src, keypoints=kp, outImage=img_src, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return kp, ds


def find_matches(des1, des2):
	# Init matcher
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)  # Find the matches

	# Get only the good matches
	good_matches = []
	for m, n in matches:
		if m.distance < 0.75 * n.distance:
			good_matches.append([m])
	...


if __name__ == '__main__':
	img1 = cv2.imread('database/Elder_Digman_Foreign_Bill_Recognition.jpg')
	img2 = cv2.imread('query/hw7_poster_2.jpg')
	find_matches(find_keypoints(img1)[1], find_keypoints(img2)[1])

