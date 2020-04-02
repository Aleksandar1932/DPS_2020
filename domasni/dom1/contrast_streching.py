"""
Aleksandar Ivanovski 186063
"""

import cv2
import matplotlib.pyplot as plt


def contrast_stretch(img, p1, p2):
    """

    :param img:
    :param p1: (x1, y1) referentna tocka 1: tuple od koordinati
    :param p2: (x2, y2) referentna tocka 2: tuple od koordinati
    :return: img so contrast stretching

    k1, k2, k3 se presmetuvaat kako koeficienti na prava soodvetno
    se iterira niz site pikseli,
    i vo zavisnost od vrednosta na pikselot mu se pravi soodvetniot stretch
    """
    k1 = (p1[1] - 0) / (p1[0] - 0)
    k2 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    k3 = (255 - p2[1]) / (255 - p2[0])

    for i in range(0, len(img)):
        for j in range(0, len(img[1])):
            if img[i][j] < p1[0]:  # < a
                img[i][j] = round(k1 * img[i][j])
            elif p1[0] <= img[i][j] <= p2[0]:
                img[i][j] = k2 * (img[i][j] - p1[0]) + p1[1]
            elif img[i][j] > p2[0]:
                img[i][j] = k3 * (img[i][j] - p2[0]) + p2[1]

    return img


if __name__ == '__main__':
    img_src = cv2.imread('sample_nature.jpeg', 0)

    gr = plt.figure()
    gr.add_subplot(2, 1, 1)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))
    gr.add_subplot(2, 1, 2)
    plt.axis('off')
    plt.title('Slika so primenet contrast stretching')
    plt.imshow(cv2.cvtColor(
        contrast_stretch(img_src, (78, 25), (56, 65)),
        cv2.COLOR_BGR2RGB))
    plt.show(block=True)
