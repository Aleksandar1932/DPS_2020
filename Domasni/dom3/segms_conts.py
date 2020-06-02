# Segmentacija na slikite
# Detekcija na konturite
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

LEAF_RATIO_TOLERANCE = 0.05
BORDER_SIZE = 15


def dir_images(dir_path):
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
    return data, files


def generate_html_report(paths):
    file = open('report.html', 'w')
    begin_html = """<html>
<head>
    <title>Segmentation and Contours Report</title>
    <style>
        img {
            display: inline-block;
            width: 25%;
        }
    </style>
</head>
<body>
<h1>Report for image structuring and leaf contour dettection</h1>
<h2>Documentation available <a href="README.md">here</a></h2>
<hr>
<h3>Aleksandar Ivanovski 186063</h3>
<br>
"""
    pic_html = ""
    for i in range(0, (int(len(paths) / 2)), 2):
        to_append = """
<div>
    <h2>Source image: {}.jpg</h2>
    <img src="{}" alt="slika">
    <img src="{}" alt="slika">
    <hr>
</div> """.format(paths[i].split('/')[1].split('_')[0], paths[i], paths[i + 1])
        pic_html += to_append
    end_html = """
</body>
</html>"""
    file.write(begin_html + pic_html + end_html)
    file.close()


if __name__ == '__main__':
    images, images_names = dir_images("source_imgs")
    result_paths = []

    img_counter = 0
    for img_color in images:
        img_color = cv2.copyMakeBorder(
            img_color,
            BORDER_SIZE,
            BORDER_SIZE,
            BORDER_SIZE,
            BORDER_SIZE,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )
        # Detekcija na struktura
        img_area = (len(img_color) * len(img_color[0]))  # Converting the BGR image to greyscale
        img_grey = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)  # Denoising the greyscale image with median blur
        img_grey = cv2.medianBlur(img_grey, 5)  # Thresholding the denoised greyscale image using OTSU's thresholding
        img_th = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Performing closing
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.dilate(cv2.erode(img_th, kernel, iterations=1), kernel, iterations=1)  # The result

        # Detekcija i iscrtuvanje na konturite
        leaf_ratio = 1 - (np.count_nonzero(np.array(closed)) / img_area)
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: abs(leaf_ratio - (cv2.contourArea(c) / img_area)))

        leaf_contour = contours[0]  # Let the first contour be the leaf contour
        cv2.drawContours(img_color, [leaf_contour], -1, (0, 0, 255), 3)

        # Rezultati zapisuvanje na disk
        image_name = images_names[img_counter].split('/')[1].split('.')[0]
        cv2.imwrite("results/{}_contour.jpg".format(image_name), img_color)
        cv2.imwrite("results/{}_structure.jpg".format(image_name), closed)

        result_paths.append("results/{}_contour.jpg".format(image_name))
        result_paths.append("results/{}_structure.jpg".format(image_name))

        # Pecatenje log
        print("=== File: {}.jpg ===\n"
              "Contour area: {}px^2\n"
              "Leaf ratio: {}\n"
              "Contour ratio: {}"
              .format(image_name,
                      cv2.contourArea(leaf_contour),
                      leaf_ratio,
                      (cv2.contourArea(leaf_contour) / img_area)))

        generate_html_report(result_paths)

        img_counter += 1
    print("===\nResults available at: report.html")