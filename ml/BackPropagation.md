### BackPropagation 反向传播

现在神经网络的参数优化，都用到Backpropagation反向传播算法，这个是神经网络的最基本的算法。

![图片](https://images2015.cnblogs.com/blog/853467/201606/853467-20160630140644406-409859737.png)

Layer L1是输入层，Layer L2是隐含层，Layer L3是隐含层，我们现在手里有一堆数据{x1,x2,x3,...,xn},输出也是一堆数据{y1,y2,y3,...,yn},现在要他们在隐含层做某种变换，让你把数据灌进去后得到你期望的输出。

BP算法的主要流程可以总结如下：
```
    输入：训练集D=(xk,yk)mk=1D=(xk,yk)k=1m; 学习率;
    过程：
        1. 在(0, 1)范围内随机初始化网络中所有连接权和阈值
        2. repeat:
        3.　　 for all (xk,yk)∈D(xk,yk)∈D do
        4. 　　　　根据当前参数计算当前样本的输出;
        5. 　　　　计算输出层神经元的梯度项；
        6. 　　　　计算隐层神经元的梯度项；
        7. 　　　　更新连接权与阈值
        8. 　　end for
        9. until 达到停止条件
    输出：连接权与阈值确定的多层前馈神经网络
```

[参考资料](https://www.cnblogs.com/charlotte77/p/5629865.html)中有简单的计算

[代码](backparopagation.py)