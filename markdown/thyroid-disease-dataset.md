### 甲状腺

### Thyroid Nodule Classification in Ultrasound Images by Fine-Tuning Deep Convolutional Neural Network
- 通过微调深度卷积神经网络在超声图像中的甲状腺结节分类
- Abstract

    检测甲状腺结节，重要的是识别尽可能多的恶性结节（malignant nodules），同时排除那些很可能是由于良性针刺活检（fine needle aspiration, FNA）活检或手术产生的良性结节。本文提出了一种计算机辅助诊断 (computer-aided diagnosis, CAD) 系统，用于对超声图像中的甲状腺结节进行分类。我们采用深度学习方法从甲状腺超声图像中提取特征。预处理超声图像以校准它们的比例并去伪（artifacts）。然后使用预处理的图像样本对预先训练的GoogLeNet模型进行微调，从未完成良好的特征提取。甲状腺超声图像的提取特征被发送到Cost-sensitive Random Forest classifier，以将图像分类为“恶性Maligns”和“良性Benigns”病例。实验结果表明，所提出的微调GoogLeNet模型实现了优异的分类性能，在开放存取数据库（Pedraza等人【16】）中获得了98.29％的分类准确度，99.10％的灵敏度和93.90％的特异性，而96.34％的分类准确度，对我们local database中的图像具有86％的灵敏度和99％的特异性。