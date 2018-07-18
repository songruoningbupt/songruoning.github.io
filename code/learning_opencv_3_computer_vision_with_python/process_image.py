# -*- coding: utf-8 -*-

"""
learning_opencv_3_computer_vision_with_python_chapter_3
- 高通滤波器
"""

import cv2
import numpy
from scipy import ndimage

kernel_3x3 = numpy.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel_5x5 = numpy.array(
    [[-1, -1, -1, -1, -1], [-1, 1, 2, 1, -1], [-1, 2, 4, 2, -1], [-1, 1, 2, 1, -1], [-1, -1, -1, -1, -1]])


def hpf_image(file):
    print("[INFO] read_image_file: " + file)
    # imread function
    image = cv2.imread(file, 0)
    print(image)
    k3 = ndimage.convolve(image, kernel_3x3)
    k5 = ndimage.convolve(image, kernel_5x5)
    # convolve 卷积

    # 高斯低通滤波器
    bluered = cv2.GaussianBlur(image, (5, 5), 0)
    g_hpf = image - bluered

    cv2.imshow("image", image)
    cv2.imshow("k3", k3)
    cv2.imshow("k5", k5)
    cv2.imshow("g_hpf", g_hpf)
    cv2.imshow("bluered", bluered)
    cv2.waitKey()
    cv2.destroyAllWindows()


def save_image(output_file, output_image):
    cv2.imwrite(output_file, output_image)


def show_image(image, name):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def blur_image(file):
    print("[INFO] read_image_file: " + file)
    # imread function
    image = cv2.imread(file, 0)
    print(image)
    ksize = 15
    # 高斯低通滤波器
    gaussianBlur = cv2.GaussianBlur(image, (ksize, ksize), 0)
    medianBlur = cv2.medianBlur(image, ksize)
    blur = cv2.blur(image, (ksize, ksize), 0)

    cv2.imshow("gaussianBlur", gaussianBlur)
    cv2.imshow("medianBlur", medianBlur)
    cv2.imshow("blur", blur)
    cv2.waitKey()
    cv2.destroyAllWindows()


def laplacian_process(file):
    print("[INFO] read_image_file: " + file)
    # imread function
    image = cv2.imread(file, 0)
    print(image)
    gray_image = cv2.Laplacian(image, cv2.CV_16S, ksize=5)
    dst = cv2.convertScaleAbs(gray_image)
    cv2.imshow("Laplacian", dst)
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def sobel_process(file):
    print("[INFO] read_image_file: " + file)
    # imread function
    image = cv2.imread(file, 0)
    print(image)

    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    cv2.imshow("absX", absX)
    cv2.imshow("absY", absY)
    cv2.imshow("Result", dst)
    cv2.imshow("image", image)
    # cv2.imshow("medianBlur", medianBlur)
    cv2.waitKey()
    cv2.destroyAllWindows()


def scharr_process(file):
    print("[INFO] read_image_file: " + file)
    # imread function
    image = cv2.imread(file, 0)
    print(image)

    x = cv2.Scharr(image, cv2.CV_16S, 1, 0)
    y = cv2.Scharr(image, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    cv2.imshow("absX", absX)
    cv2.imshow("absY", absY)
    cv2.imshow("Result", dst)
    cv2.imshow("image", image)
    # cv2.imshow("medianBlur", medianBlur)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("learning_opencv_3_computer_vision_with_python_chapter_2")
    image_file = "../../image/test.jpg"
    # hpf_image(image_file)
    # blur_image(image_file)
    # laplacian_process(image_file)
    # sobel_process(image_file)
    scharr_process(image_file)
