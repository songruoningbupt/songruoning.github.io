## 处理文件、摄像头和图形用户界面

### 基本IO操作

#### 读写图像文件

- imread()和imwrite()能支持各种静态文件格式 *在[python code](https://github.com/songruoningbupt/thyroid_nodule/blob/master/code/src/preprocess/read_xml.py)中有尝试*
    - 可以读一种格式的，然后存成其他格式的
    - 默认情况下，imread会返回BGR格式的图像，下面的参数可以作为选项
        - IMREAD_ANYCOLOR = 4
        - IMREAD_ANYDEPTH = 2
        - IMREAD_COLOR = 1
        - IMREAD_GRAYSCALE = 0
        - IMREAD_LOAD_GDAL = 8
        - IMREAD_UNCHANGED = -1
            ```
                image = cv2.imread(file, 0)
                print(image)
                # [[0  0  0...  5  1  2]
                #  [0  0  0...  5 15  0]
                #  [0  0  0...  0  5  1]
                #  ...
                #  [0  0  0...  0  0  0]
                #  [0  0  0...  0  0  0]
                #  [0  0  0...  0  0  0]]
            ```
- 在Python和Numpy中表示一副图像的细节 *无论哪个格式，每个像素都会有一个值，但是不同格式表示像素的方式有所不同*
    - 通过二维NumPy数组来简单的创建一个黑色的正方形图像，每一个像素都由一个8位整数来表示
        ```
            def numpy_image():
                img = numpy.zeros((3, 3), dtype=numpy.uint8)
                print(img)
                # [[0 0 0]
                #  [0 0 0]
                # [0 0 0]]
        ```
    - 通过cv2.cvtColor来添加上颜色
        ```
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
        ```
    - 每个像素都是有3元数组表示，即为BGR通道，同理的还有HSV
    - img.shape可以得到图像结构，会返回行和列(3, 3)，BGR格式的，还会有通道数(3, 3, 3)