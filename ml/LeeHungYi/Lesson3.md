### Back Propagation

http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/DNN%20backprop.ecm.mp4/index.html

- background
    - Cost Function
    - Gradient Descent
- Chain Rule
    - Case 1： x的变动会影响y的变动，也会影响z的变动
        ![image](http://latex.codecogs.com/gif.latex?y%3Dg%28x%29) [image](http://latex.codecogs.com/gif.latex?z%3Dh%28y%29)
        ![image](http://latex.codecogs.com/gif.latex?%5CDelta%20x%20%5Cto%20%5CDelta%20y%20%5Cto%20%5CDelta%20z) [image](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20x%7D%3D%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y%7D%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D)
    - Case 2： s能通过x去影响z，也能通过y来影响z
        ![image](http://latex.codecogs.com/gif.latex?x%3Dg%28s%29)
        ![image](http://latex.codecogs.com/gif.latex?y%3Dh%28s%29)
        ![image](http://latex.codecogs.com/gif.latex?z%3Dk%28x%2Cy%29)
        ![image](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20s%7D%3D%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20x%7D%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20s%7D&plus;%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y%7D%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20s%7D)
- Back Propagation算法公式推导
    ![image](http://latex.codecogs.com/gif.latex?%5Cpartial%20C%5E%7Br%7D%20/%20%5Cpartial%20w_%7Bij%7D%5E%7Bl%7D)
    - 其中，First Term 第一项，其实就是前一级的输出
        ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/firstterm.jpg)
    - Second Term 第二项
        ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/secondterm.jpg)
        - How to compute δ，可以分为两个部分，一个是y对z求偏导，输出是激励函数的导数
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/secondterm1.jpg)
        - The relation of δ(l) and δ(l+1) 结果类似于一个逆向的新的神经网络
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/secondterm2.jpg)
    - Conclude
            ![image](https://github.com/songruoningbupt/songruoningbupt.github.io/blob/master/image/LeeHungYi/secondterm3.jpg)
