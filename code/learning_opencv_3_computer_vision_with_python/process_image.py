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
    cv2.waitKey()
    cv2.destroyAllWindows()


def canny_process(file):
    # Canny边缘检测
    print("[INFO] read_image_file: " + file)
    image = cv2.imread(file, 0)
    print(image)
    #    cv2.imwrite("../../image/learning_opencv_3_computer_vision_with_python/test_canny.jpg", cv2.Canny(image, 200, 300))
    cv2.imshow("Canny", cv2.Canny(image, 80, 300))
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def contour_process(file):
    # 轮廓检测
    print("[INFO] read_image_file: " + file)
    # image = numpy.zeros((200, 200), dtype=numpy.uint8)
    # image[50:150, 50:150] = 255
    image = cv2.imread(file, 0)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    # findContours()有三个参数：输入图像、层次类型和轮廓逼近方法
    #
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    image = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)

    # cv2.imwrite("../../image/learning_opencv_3_computer_vision_with_python/test_contour.jpg", image)
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("learning_opencv_3_computer_vision_with_python_chapter_3")
    image_file = "../../image/test.jpg"
    image_file = "../../image/1_1.jpg"
    # hpf_image(image_file)
    # blur_image(image_file)
    # laplacian_process(image_file)
    # sobel_process(image_file)
    # scharr_process(image_file)
    # canny_process(image_file)
    contour_process(image_file)
