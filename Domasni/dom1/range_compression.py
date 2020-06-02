"""
Aleksandar Ivanovski 186063
"""

import cv2
import matplotlib.pyplot as plt
import math


def range_compression(img, c):
    """
    :param img:
    :param c:
    :return: img so range compression
    """
    for i in range(0, len(img)):
        for j in range(0, len(img[1])):
            img[i][j] = c * math.log10(1 + img[i][j])
    return img


if __name__ == '__main__':
    img_src = cv2.imread('sample_nature.jpeg', 0)

    gr = plt.figure()
    gr.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.title('Original:')
    plt.imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))
    gr.add_subplot(1, 2, 2)
    plt.axis('off')
    plt.title('Range Compression:')
    plt.imshow(cv2.cvtColor(
        range_compression(img_src, 80),
        cv2.COLOR_BGR2RGB))
    plt.show(block=True)
