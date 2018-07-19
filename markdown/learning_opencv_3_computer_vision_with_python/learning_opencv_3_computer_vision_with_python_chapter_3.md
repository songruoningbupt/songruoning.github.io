### 使用OpenCV 3处理图像

*自己的[code](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/code/learning_opencv_3_computer_vision_with_python/process_image.py)*

#### 不同色彩空间的转换

- 灰度
- BGR
- HSV，Hue色调，Saturation饱和度，Value黑暗程度

#### 傅立叶变换

##### 高通滤波器HPF与低通滤波器LPF

*HPF是检测图像的某个区域，然后根据像素与周围像素的亮度差值来提升(boost)该像素的亮度的滤波器*

*LPF是检测像素与周围像素的亮度差值小于一个特定值时，平滑该像素的亮度，主要用于去燥和模糊化*

##### 边缘检测

OpenCV提供了许多边缘检测滤波器，包括
- Laplacian()：拉普拉斯变换
    - `dst = cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])`
    - src 原图像
    - ddepth 图像的深度， -1 表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度
    - dst 目标图像 【可选参数】
    - ksize 算子的大小，必须为1、3、5、7。默认为1 【可选参数】
    - scale 是缩放导数的比例常数，默认情况下没有伸缩系数 【可选参数】
    - delta 是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中 【可选参数】
    - borderType 是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。。 【可选参数】
- Sobel()：离散微分算子 (discrete differentiation operator)。 它结合了高斯平滑和微分求导，用来计算图像灰度函数的近似梯度。
    - `dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])`
    - 第一个参数是需要处理的图像；
    - 第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
    - dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2
    - dst不用解释了；
    - ksize是Sobel算子的大小，必须为1、3、5、7。
    - scale是缩放导数的比例常数，默认情况下没有伸缩系数；
    - delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
    - borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。

    在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口。convertScaleAbs()的原型为：`dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])`,其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来。其函数原型为：`dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])`,其中alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值。
- Scharr()：也用于检测水平和垂直方向的图像的二阶导数。

##### 用定制内核做卷积

核是一组权重，它决定如何通过邻近像素点来计算新的像素点，核也被称作卷积矩阵，它对一个区域的像素做调和（mix up）或卷积运算。通常基于核的滤波器叫做卷积滤波器。

卷积矩阵是一个二维数组，有奇数行、奇数列，中心的元素对应于感兴趣的像素，其他元素对应于这个像素周围的邻近像素，每个元素都有一个整数或浮点数的值，这些值就是应用在像素值上的权重。

OpenCV提供了filter2D()来处理童虎指定的任意卷积矩阵。filter2D()会对每一个通道都用同样的核，如果通道使用不同的核，就必须用split()和merge()

##### Canny边缘检测

![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/learning_opencv_3_computer_vision_with_python/test_canny.jpg)

##### 轮廓检测

![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/learning_opencv_3_computer_vision_with_python/test_contour.jpg)

##### 边界框、最小矩形区域和最小闭圆的轮廓

##### 检测形状

