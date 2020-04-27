import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

kernels = [
    np.array(([1, 1, 0], [1, 0, -1], [0, -1, -1]), dtype="float32"),
    np.array(([1, 1, 1], [0, 0, 0], [-1, -1, -1]), dtype="float32"),
    np.array(([0, 1, 1], [-1, 0, 1], [-1, -1, 0]), dtype="float32"),
    np.array(([1, 0, -1], [1, 0, -1], [1, 0, -1]), dtype="float32"),
    # Ostanatite 4 kerneli moze da se dobijat so mnozenje so -1 na ovie matrici
    # no zaradi poednostavuvanje ke bidat harkodirani

    np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]), dtype="float32"),
    np.array(([-1, -1, -1], [0, 0, 0], [1, 1, 1]), dtype="float32"),
    np.array(([0, -1, -1], [1, 0, -1], [1, 1, 0]), dtype="float32"),
    np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype="float32"),
]


def apply_filters(src_img):
    """

    :param src_img:
    :return: Lista od 8 slika i vrz sekoja slika e apliciran po eden od 8 te kerneli
    """
    return [cv2.filter2D(src_img, -1, kernel) for kernel in kernels]


def show_all_filtered_images(f_imgs):
    k_c = 1
    for f_img in f_imgs:
        cv2.imshow("Source image filtered with Kernel #{}".format(k_c), f_img)
        cv2.waitKey(0)
        k_c += 1


def calculate_result(f_imgs, shape):
    flatten_imgs = []
    for f_img in f_imgs:
        flatten_imgs.append(f_img.flatten())

    return np.reshape(np.maximum.reduce(flatten_imgs), shape)


def threshold_edges(img, min_value):
    return cv2.threshold(img, min_value, 255, cv2.THRESH_BINARY)[1]


if __name__ == '__main__':
    src_image = cv2.imread("lena.png", 0)
    filteredImgs = apply_filters(src_image)  # gi aplicirame 8-te

    # Prikaz na poedinecni aplikacii na sekoj kernel
    show_all_filtered_images(filteredImgs)
    img_edges = calculate_result(filteredImgs, filteredImgs[0].shape)

    # Prikaz na Originalnata slika i detektiranite rabovi
    cv2.imshow("Original Image", src_image)
    cv2.imshow("Detected Edges", img_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Thresholding na detektiranite rabovi
    def on_trackbar(val):
        cv2.imshow("Thresholding Demo", threshold_edges(img_edges, val))


    cv2.namedWindow("Thresholding Demo")
    cv2.createTrackbar("Lower Threshold Value:", "Thresholding Demo", 0, 255, on_trackbar)
    on_trackbar(0)
    cv2.waitKey(0)
