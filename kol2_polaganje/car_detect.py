import cv2
import numpy as np

if __name__ == '__main__':
	car_bgr = cv2.imread('car.jpg')
	car_grey = cv2.cvtColor(car_bgr, cv2.COLOR_BGR2GRAY)

	car_grey = cv2.medianBlur(car_grey, 3)

	car_grey_thresh = cv2.threshold(car_grey, 0, 255, cv2.THRESH_OTSU)[1]

	kernel = np.ones((5, 5), np.uint8)
	closed = cv2.dilate(cv2.erode(car_grey_thresh, kernel, iterations=1), kernel, iterations=2)

	contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

	for c in contours:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.1 * peri, True)

	cv2.drawContours(car_bgr, [contours[3]], -1, (0, 0, 255), 3)

	x, y, w, h = cv2.boundingRect(contours[3])
	plate = car_bgr[y:y + h, x:x + w]

	cv2.imwrite("cc", plate)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
