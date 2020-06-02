import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    plan = cv2.imread("scanned_plans/Vasilevo 71.jpg", 0)

    ret, thresh1 = cv2.threshold(plan, 183, 255, cv2.THRESH_BINARY)

    # cv2.imshow("Original", thresh1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("scanned_plans/scanned_grey.jpg", plan)
    cv2.imwrite("scanned_plans/trsh.jpg", thresh1)
