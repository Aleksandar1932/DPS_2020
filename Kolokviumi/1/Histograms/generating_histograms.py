import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("../SampleImages/parrot.jpg", 0)
    hist = cv2.calcHist([img], [0], None, [255], [0, 256])

    plt.plot(hist, color="gray")
    plt.xlim([0, 256])
    plt.show()

