import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("../SampleImages/parrot.jpg", 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    claheImg1 = clahe.apply(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    claheImg2 = clahe.apply(img)

    cv2.namedWindow("main", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    cv2.imshow("8x8 Grid", claheImg1)
    cv2.waitKey(0)

    cv2.imshow("32x32 Grid", claheImg2)
    cv2.waitKey(0)
