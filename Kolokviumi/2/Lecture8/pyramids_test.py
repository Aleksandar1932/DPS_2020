import cv2
import numpy as np

if __name__ == '__main__':
	img = cv2.imread('TemplatesPyramids/messi5.jpg')
	pics = []

	for i in range(0, 5):
		img = cv2.pyrDown(img)
		pics.append(img)

	for ind,img in enumerate(pics):
		cv2.imshow(f'{ind}', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
