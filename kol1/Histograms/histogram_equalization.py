import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("../SampleImages/parrot.jpg", 0)
    eqImg = cv2.equalizeHist(img)

    cv2.imshow("Original image", img)
    cv2.waitKey(0)

    cv2.imshow("Equalized", eqImg)
    cv2.waitKey(0)


