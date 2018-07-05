### What is Machine Learning, Deep Learning and Structured Learning?

http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/Brief%20ML%20(v2).ecm.mp4/index.html

- 怎么让机器学习 Looking for a Function
    - 语音识别 就是 找到一个语音识别的Function
    - 。。。
- Framework
    - Training Data： <x,y> x:function input y:function output
    - Model: Functiond的集合
    - Train： Pick the best function
- What is Deep Learning
    - 生产线 Hypothesis Function：
        Simple Function 1 -> Simple Function 2 -> ... -> Simple Function  become  A very complex funciton
    - A very complex funciton is produced By Machine itself
    - Deep learning 通常指基于神经网络的方法
- A Neuron for Machine
    - 每一个神经元就是一个非常简单的Function
    - Activation function： Sigmoid function
- Why Deep Learning？ Deeper is Better
    - 深度效果更好，因为它使用更多参数
    - “矮胖型”，只用一层Hidden Layer vs “高瘦型” Deeper，“高瘦型”需要更少的parameter，所以需要的更少的train learning，performance更好，矩阵更小