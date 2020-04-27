import cv2
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("../SampleImages/flower.jpg", 0)

    img = cv2.resize(img, (int(img.shape[0] * 0.2), int(img.shape[1] * 0.2)))

    orig_size = img.shape
    flat_x = img.flatten()

    clipped_img = np.clip(flat_x, 50, 150).reshape(orig_size)

    cv2.imshow("Original", img)
    cv2.waitKey(0)

    cv2.imshow("Clipped", clipped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()