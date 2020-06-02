import cv2
import numpy as np

if __name__ == '__main__':
	img_bgr = cv2.imread("example_img.jpg")
	img_grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((10, 10), np.uint8)
	img_grey_er = cv2.erode(src=img_grey, kernel=kernel, iterations=1)
	img_grey_dl = cv2.dilate(src=img_grey, kernel=kernel, iterations=1)
	img_grey_dl_trsh = cv2.threshold(img_grey_dl, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	img_grey_trsh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

	img_grey_dl_edges = img_grey - img_grey_dl
	img_grey_dl_edges_blur =  cv2.medianBlur(img_grey_dl_edges, 5)

	cv2.imshow("Image Eroded", img_grey_er)
	cv2.imshow("Image Dilated", img_grey_dl)
	cv2.imshow("Image Dilated&Thresholded", img_grey_dl_trsh)
	cv2.imshow("Image Greyscale", img_grey)
	cv2.imshow("Image Dilated Edges", img_grey_dl_edges)
	cv2.imshow("Image Dilated Edges w/ Blur", img_grey_dl_edges_blur)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
