### Deep Learning More Techniques

http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/Deep%20More%20(v2).ecm.mp4/index.html

- 激活函数
    - ReLU函数 Rectified Linear Unit （2011年的Paper）
        ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/relu.jpg)
         - Fast to Compute 计算快
         - Biological reason 生物学原因
         - Infinite sigmoid with different biases 无限叠加的情况下，就很像ReLU函数（need to survey）
    - sigmoid函数的问题
        ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/problemofsigmoid.jpg)
        - 绿色的是sigmoid的曲线，蓝色的是sigmoid的微分，发现sigmoid的微分都是<1，最大是0.25
        - Error signal is getting smaller and smaller, Gradient is smaller
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/vanishingGradientProblem.jpg)
        - 前项传导，在多层隐藏层中，会发现前面的权值学习会很慢，而后面的学习相应的更快一些
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/vanishingGradientProblem1.jpg)
    - ReLU函数
        - 如果使用ReLU的时候，微分不是0就是1，微分为0的weight值，从正向神经网络来看，该神经元对应的上一级output也等于0，所以就可以当做不存在，可以忽略掉影响小的输入，让神经网络变瘦
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/relubackward.jpg)
    - Maxout
        - ReLU is special case of Maxout，Maxout可以通过一定的参数变成ReLU
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/reluisspecialcase.jpg)
        - Maxout能做比ReLU更灵活的事情，Learnable Activation Function，可以根据数据学出来
        - 未完，等我学学别的

- Cost Funtion 评价函数，说明这个分类器差距有多大
    - Softmax：竟然没听懂，待Survey

- Optimization
    - Vanilla Gradient Descent 一般
        - How to determine the learning rates
        - Stuck at local minima or saddle points
    - learning rates
        - 偏大或者偏小，都会影响Learning rates和performance
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/learningRates.jpg)
        - 简单有效的方法：随着每一个epoch减小参数，来降低learning rate
        - Adagrad：每一个参数的learning rate都应该要除掉它之前的微分值的均方根
            - Vanilla Gradient descent gt是第t层的微分 （特点：gradient越大，步伐越大）
                ![formula](http://latex.codecogs.com/gif.latex?w%5E%7Bt&plus;1%7D%5Cleftarrow%20w%5E%7Bt%7D-%5Ceta%20%5E%7Bt%7Dg%5E%7Bt%7D)
                ![formula](http://latex.codecogs.com/gif.latex?g%5E%7Bt%7D%3D%5Cfrac%7B%5Cpartial%20C%28%5Ctheta%20%5E%7Bt%7D%29%7D%7B%5Cpartial%20w%7D)
                ![formula](http://latex.codecogs.com/gif.latex?%5Ceta%20%5E%7Bt%7D%3D%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bt&plus;1%7D%7D)
            - Adagrad
                ![formula](http://latex.codecogs.com/gif.latex?w%5E%7Bt&plus;1%7D%5Cleftarrow%20w%5E%7Bt%7D-%5Cfrac%7B%5Ceta%20%5E%7Bt%7D%7D%7B%5Csigma%20%5E%7Bt%7D%7Dg%5E%7Bt%7D)
                ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/adagrad.jpg)
            - 随着epoch的增加，分子越来越小，分母会越来越大，所以learning rate会越来越小
                ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/adagrad1.jpg)
        - Why lagrger gradient, larger steps 关于步伐，改变的步伐应该与目标点的距离成正比，越接近越慢才对，而随之参数的增多，一般的梯度下降算法不太能满足
        - Adagrad可以优化
    - Stuck at local minima or saddle points
        - Momentum：movement of last step minus gradient at present 上一步的方向加上这一步梯度的负方向 （类似于惯性）
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/momentum.jpg)

- Generalization
    - Panacea: 有更多的测试集，生成更多的测试集
    - Early Stop
    - Weight Decay
    - Dropout



