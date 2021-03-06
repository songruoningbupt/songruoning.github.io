# -*- coding: utf-8 -*-

"""
learning_opencv_3_computer_vision_with_python_chapter_2
- 基本IO操作
"""

import cv2
import numpy
import os


def read_image(file):
    print("[INFO] read_image_file: " + file)
    # imread function
    image = cv2.imread(file)
    print(image)
    # image = cv2.imread(file, 0)
    # print(image)
    # [[0  0  0...  5  1  2]
    #  [0  0  0...  5 15  0]
    #  [0  0  0...  0  5  1]
    #  ...
    #  [0  0  0...  0  0  0]
    #  [0  0  0...  0  0  0]
    #  [0  0  0...  0  0  0]]
    show_image(image)


def save_image(output_file, output_image):
    cv2.imwrite(output_file, output_image)


def show_image(image):
    cv2.namedWindow("Image")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def numpy_image():
    img = numpy.zeros((300, 300), dtype=numpy.uint8)
    print(img)
    # [[0 0 0]
    #  [0 0 0]
    # [0 0 0]]
    # 输出纯黑图像
    # show_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print(img)
    # [[[0 0 0]
    #  [0 0 0]
    # [0 0 0]]
    # [[0 0 0]
    # [0 0 0]
    # [0 0 0]]
    # [[0 0 0]
    # [0 0 0]
    # [0 0 0]]]
    show_image(img)


def image_to_array():
    img = numpy.zeros((300, 300), dtype=numpy.uint8)
    print(img)
    img[100, 100] = 255
    show_image(img)
    img = numpy.zeros((300, 300), dtype=numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[100, 100, 0] = 255
    print(img[100,100])
    show_image(img)


def random_image():
    randomByteArray = bytearray(os.urandom(120000))
    flatNumpyArray = numpy.array(randomByteArray)
    grayImage = flatNumpyArray.reshape(300, 400)
    show_image(grayImage)
    bgrImage = flatNumpyArray.reshape(100, 400, 3)
    show_image(bgrImage)


if __name__ == '__main__':
    print("learning_opencv_3_computer_vision_with_python_chapter_2")
    image_file = "../../image/thyroid/1_1_label.jpg"
    # read_image(image_file)
    # numpy_image()
    # image_to_array()
    random_image()
