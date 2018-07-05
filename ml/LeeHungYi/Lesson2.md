### Neural Network (Basic Ideas)

http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/DNN%20(v4).ecm.mp4/index.html

- Classification主要分为
    - Binary Classification 2分类
    - Multi-class Classification eg: Handwriting Digit Classification, Image Recognition
- Single Neuron
    ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/SingleNeuron.jpg)
    - 单神经元只能做2分类，不能做多分类
    - Limitation of Single Layer
        - 做不到简单的XOR，但是Neural Network可以
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/xor_neural_network.jpg)
            途中a1和a2就是隐藏层
- Neuron Network
    - l层输入z与前一层输出a的关系
        ![formula1](http://latex.codecogs.com/gif.latex?z%5E%7Bl%7D%3DW%5E%7Bl%7Da%5E%7Bl-1%7D&plus;b%5E%7Bl%7D)
        ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/RelationsBetweenLayerOutput1.jpg)
    - l层输出a与l层输入z的关系，其中σ是指Activation function（常用Sigmoid function）
        ![formula1](http://latex.codecogs.com/gif.latex?a%5E%7Bl%7D%3D%5Csigma%20%28z%5E%7Bl%7D%29)
        ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/RelationsBetweenLayerOutput2.jpg)
    - l层输出a与l-1层输出a的关系
        ![formula](http://latex.codecogs.com/gif.latex?a%5E%7Bl%7D%3D%5Csigma%20%28W%5E%7Bl%7Da%5E%7Bl-1%7D&plus;b%5E%7Bl%7D%29)
    - 所以神经网络的Function就是
        ![formula](http://latex.codecogs.com/gif.latex?y%3Df%28x%29%3D%5Csigma%20%28W%5E%7BL%7D...%5Csigma%20%28W%5E%7B2%7D%5Csigma%20%28W%5E%7B1%7Dx&plus;b%5E%7B1%7D%29&plus;b%5E%7B1%7D%29&plus;b%5E%7B1%7D%29)
    - Best Function = Best Parameters 不同的参数产生不同的functiong，所以最好的Function就是挑选最好的参数
        ![formula](http://latex.codecogs.com/gif.latex?define%3A%20f%28x%3B%5Ctheta%20%29%20-%3E%20parameter%20set)
        ![formula](http://latex.codecogs.com/gif.latex?%5Ctheta%20%3D%5Cleft%20%5C%7B%20W%5E%7B1%7D%2Cb%5E%7B1%7D%2CW%5E%7B2%7D%2Cb%5E%7B2%7D...W%5E%7BL%7D%2Cb%5E%7BL%7D%20%5Cright%20%5C%7D)
- Cost Function
    - C(θ): cost/loss/error function 来衡量一组参数有多坏
    - O(θ): 来衡量一组参数有多好
    - Gradient Descent：local minima 局部最小值？  saddle point 马鞍点？
- Practical Issues for neural network
    - Parameter Initialization, Suggest
         - 不要把Parameter都设置成一样的
         - 设置Parameter时要随机一点
    - Learning Rate
         - Set the learning rate η carefully，太大可能飞出去或者不能下降到0，太小可能影响performance
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/learningrate.jpg)
