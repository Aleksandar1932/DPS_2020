import cv2
import numpy as np
import os
import glob

BORDER_SIZE = 15


def get_leaf_contour(img_leaf):
    # Funkcija koja kako argument prima slika so list, primenuva tehniki za segmentacija i kako rezultat
    # ja vraka konturata na listot vo sodoveten format podrzhan od opencv2.

    """
    :param img_leaf:
    :return: leaf_contour, see open-cv contours
    """

    img_leaf = cv2.copyMakeBorder(
        img_leaf,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )
    img_area = (len(img_leaf) * len(img_leaf[0]))  # Converting the BGR image to greyscale
    img_grey = cv2.cvtColor(img_leaf, cv2.COLOR_BGR2GRAY)  # Denoising the greyscale image with median blur
    img_grey = cv2.medianBlur(img_grey, 5)  # Thresholding the denoised greyscale image using OTSU's thresholding
    img_th = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Performing closing
    kernel = np.ones((4, 4), np.uint8)
    closed = cv2.dilate(cv2.erode(img_th, kernel, iterations=1), kernel, iterations=1)  # The result

    # Detekcija i iscrtuvanje na konturite
    leaf_ratio = 1 - (np.count_nonzero(np.array(closed)) / img_area)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: abs(leaf_ratio - (cv2.contourArea(c) / img_area)))

    leaf_contour = contours[0]  # Let the first contour be the leaf
    cv2.drawContours(img_leaf, [leaf_contour], -1, (0, 0, 255), 3)
    return leaf_contour


#
# def map_leaf_contour(imgs):
#     # Funkcija koja kako argument prima lista od sliki, za sekoja slika ja odreduva konturata na listot
#     # Mapira slika vo kontura na list vo recnik
#     """
#
#     :param list of imgs:
#     :return: dict img->leaf contour
#     """
#     contours_dict = {}
#     for img in imgs:
#         contours_dict[hash(img.tobytes())] = get_leaf_contour(img)
#     return contours_dict
#
#
# def query_img(query_cont, database_conts):
#     results = {}
#     for img, cont in database_conts.items():
#         results[img] = cv2.matchShapes(query_cont, cont, 1, 0.0)
#     return results


def open_imgs(dir_path):
    """
    :param dir_path:
    :return: list of images
    """
    img_dir = "{}".format(dir_path)  # Enter Directory of all images
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = cv2.imread(f1)
        data.append(img)
    return data


if __name__ == '__main__':
    query_imgs = open_imgs('query')
    query_imgs_dict = {}
    for img in query_imgs:
        query_imgs_dict[hash(img.tobytes())] = img

    database_imgs = open_imgs('database')
    database_imgs_dict = {}
    for img in database_imgs:
        database_imgs_dict[hash(img.tobytes())] = img

    query_conts = {}
    for k, v in database_imgs_dict.items():
        query_conts[k] = get_leaf_contour(v)

    database_conts = {}
    for k, v in database_imgs_dict.items():
        database_conts[k] = get_leaf_contour(v)

    print(database_conts.values())

    QUERY_IMG = query_imgs[0]

    print(query_imgs(query_conts[QUERY_IMG.tobytes()], database_conts))
