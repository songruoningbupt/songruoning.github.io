### anaconda3 + Pycharm的配置

*20180713，因为要做一些python的工作，所以要在windows搭建一套anaconda3 + pycharm的开发环境*

*20180809，在PyCharm中配置了tensorflow的使用*

- anaconda
    - Conda是一个开源的包、环境管理器，可以用于在同一个机器上安装不同版本的软件包及其依赖，并能够在不同的环境之间切换
    - Anaconda包括Conda、Python以及一大堆安装好的工具包，比如：numpy、pandas等
    - Miniconda包括Conda、Python
- 安装
    - https://www.anaconda.com/download/ 下载 Anaconda Python 3.6 version，建议直接上3，不用2
    - 按照步骤逐步安装
- 安装Pycharm
- 在Pycharm的File>>settings>>Project Interpreter>>Add local  里面添加Anaconda python.exe. 应用之后就可以调用各种Anaconda的库啦
- 配置完pycharm调用Anaconda后，可以快乐的在pycharm里面调用各种科学计算库

### 安装LabelMe

- 打开anaconda prompt
- 执行conda create --name=labelme python=3.6 （这一步python=*选择自己的Python版本），遇到Proceed([y]/n)?时，输入y
- activate labelme
- conda install pyqt, 遇到Proceed([y]/n)?时，输入y
- pip install labelme

### 打开LabelMe
- activate labelme
- labelme

![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/labelme.png)

### 配置TensorFlow

- 可以像labelme一样通过anaconda去解决
- 我直接在PyCharm中解决，这样方便，不需要每次都通过activate来切换环境
    - 在Pycharm的File>>settings>>Project Interpreter>>Add local  里面配置Conda Environment中的conda.exe为对应Anaconda的conda.exe文件
        ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/condaenvironment.png)
    - Apply之后，就可以添加tensorflow的Package
        ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/tensorflowPackage.png)
    - 测试Tensorflow
        ```
            import tensorflow as tf
            hello = tf.constant("Hello!TensorFlow")
            sess = tf.Session()
            print(sess.run(hello))
        ```
        ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/tensorflowtest.png)

