import glob

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

	return tuple(good_matches)


def open_imgs(dir_path):
	"""
	:param dir_path:
	:return: list of images
	"""
	img_dir = "{}".format(dir_path)  # Enter Directory of all images
	data_path = cv2.os.path.join(img_dir, '*g')
	files = glob.glob(data_path)
	data = []
	for f1 in files:
		img = cv2.imread(f1)
		img = cv2.resize(img, (int(img.shape[1] * 0.2), int(img.shape[0] * 0.2)), 0.2, 0.2)
		data.append(img)
	return data


def get_img_hash(img):
	return repr(hash(img.tobytes()))


def describe_posters(posters):
	kps_and_dsc = {}
	for poster in posters:
		kps_and_dsc[get_img_hash(poster)] = find_keypoints(poster)

	return kps_and_dsc


def index_posters(posters):
	p_index = {}
	for poster in posters:
		p_index[get_img_hash(poster)] = poster
	return p_index


if __name__ == '__main__':
	posters = open_imgs("database")
	posters_index = index_posters(posters)
	poster_kps_and_dsc = describe_posters(posters)

	query_img = cv2.imread('query/hw7_poster_1.jpg')
	query_img_resize = img = cv2.resize(query_img, (int(query_img.shape[1] * 0.2), int(query_img.shape[0] * 0.2)), 0.2,
										0.2)

	kp, ds = find_keypoints(query_img_resize)

	all_matches = {}

	for (poster_hash, poster_kps_dsc) in poster_kps_and_dsc.items():
		all_matches[poster_hash] = find_matches(poster_kps_dsc[1], ds)

	# max(all_matches, len(all_matches.get))
	cv2.imshow("a",
			   posters_index[sorted([(kv[0], len(kv[1])) for kv in all_matches.items()], key=lambda x: x[1])[-1][0]])
	cv2.waitKey(0)
	cv2.destroyAllWindows()
