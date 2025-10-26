---
layout: single
title: "神经网络综述"
date: 2025-10-23
categories:
  - 人工智能
  - 深度学习
tags:
  - 神经网络
  - 综述
  - CNN
  - RNN
toc: true
toc_label: "目录"
toc_sticky: true
header:
  teaser: /assets/images/neural-network-review/neural-network-teaser.jpg
---

-- 文档内容为学习记录，整理来源于网络，仅用于个人学习

> 本文档共同整理者：zbr1234 

## 一、人工神经网络的发展

人工神经网络（Artificial Neural Networks，ANN）是一种可用于处理具有多个节点和多个输出点的实际问题的网络结构。虽然人类的大脑和人工神经网络的运用都具有极其强大的信息处理能力，但是两者还是有许多不同之处。谷歌Deepmind最初被Demis Hassabis、Mustafa Suleyman 以及 Shane Legg 创立出来，在 2016年创造出 AlphaGo 打败世界围棋冠军李世石后逐渐被人认可，也说明人工神经网络具有巨大的潜力。与人脑处理信息方式有所不同，运用人工神经网络开发出的机器人采用线性的思维方式处理获取到的信息，计算机通过快速、精确的顺序数值运算，在串行算术类型的任务处理上超过人类。但人脑的"并行处理体系"相对于人工神经网络领域具有绝对领先的能力。

- McCulloch心理学家和Pitts数学家于1943年考虑寻找神经元背后的基本原理，将阈值函数作为计算神经元的主要特性，把逻辑演算表述为神经计算架构，提出"神经网络"概念和M-P模型，标志着人工神经网络ANN萌芽。
- Hebb 假设突触权重的变化会如何控制神经元相互激励的方式，在 1949 年出版的《行为的组织》中提出了Hebb突触以及Hebb学习规则，为人工神经网络算法的发展构建了理论知识基础。
- 20 世纪 60 年代末，Rosenblatt 开创了感知器，感知器是建立在 M-P 模型基础上，第一个物理构建并形成了具有学习能力的人工神经网络。
- 1984 年 Hopfield 神经网络（Hopfiled Neural Network，HNN）首次引入，从此基于Hopfield 神经网络的动力学行为的理解应用于信息处理和工程研究起到了至关重要的作用。
- 反向传播网络（Backpropagation Neural Network，BPNN）之后被提出用于解决多层神经网络所反应出来的问题，但是BP网络仍存在一部分缺点，比如：收敛速度慢以及大样本数据难以收敛，容易出现局部最小化。
- 1998年Lecun等基于福岛邦彦提出的卷积和池化网络结构，将 BP 算法运用到该结构的训练中，形成了卷积神经网络（Convolutional Neural Network，CNN）的雏形 LeNet-5。

随着BP算法、遗传算法、模糊神经网络等的发明，以及电脑科学技术、大数据分析、人工智能的发展，让人工神经网络步入了稳步发展时代，并且渐渐与各个学科领域结合。发展让人工神经网络进入了稳步发展时期，并且渐渐与各个学科领域结合。

本文针对人工神经网络领域中的几个模型（多层感知器（Multilayer Perceptron，MLP）、反向传播神经网络、卷积神经网络、递归神经网络（Recursive Neural Network，RNN））基本结构进行介绍，并对其相对热门的应用进行简单的概述。

## 二、算法的发展

### 2.1 前馈神经网络（FNN）

#### 2.1.1 多层感知器（MLP）

多层感知器，又称为多层前馈神经网络，如图 ，具有出色的非线性匹配和泛化能力。训练MLP使用反向传播算法，可以减少 MLP 输出数据与实际所需数据之间的全局误差。

![多层感知器结构]({{ "/assets/images/neural-network-review/image-20250306103142914.png" | relative_url }})

**工作原理：** 多层感知器（MLP）是最基础的前馈神经网络，由输入层、多个隐藏层和输出层组成。每一层的神经元与下一层的所有神经元全连接，通过激活函数引入非线性。

- sigmoid 函数

![sigmoid函数]({{ "/assets/images/neural-network-review/sigmoid-function.jpeg" | relative_url }})

- **优缺点：**

于MLP 具有非常好的非线性映射能力、较高的并行性以及全局优化的特点，现今在图像处理、预测系统、模式识别等方面取得了不错的成就。尽管 MLP 架构具有很多优点，但在高维空间下的效率相对低下，可能导致模型训练中过拟合的情况。并且由于隐藏层的存在加大了超参数的数量，使得训练过程中在收敛缓慢的情况下需要处理很高的计算量。传统的MLP实值模型中单个神经元能接收的数据输入为单个实数，在其进行多维信号输入时，通常达不到令人满意的效果。

| 特点     | 优点                                     | 缺点                                            |
| -------- | ---------------------------------------- | ----------------------------------------------- |
| 结构简单 | 易于理解和实现                           | 难以处理高维数据和复杂模式                      |
| 通用性强 | 可用于回归和分类任务                     | 容易过拟合，尤其在深层网络中                    |
| 可扩展性 | 可以通过增加隐藏层和神经元数量来提升表现 | 计算量随着网络深度和宽度迅速增加<br/>应用案例： |

**应用案例：**

- 手写数字识别： 使用MNIST数据集进行数字分类。
- 基本回归任务： 预测房价、股票价格等。

**改进算法：**

- García-Pedrajas等人提出一种广义多层感知器（Generalized Multilayer Perceptron，GMLP）的协同进化。模型基于模块的不同子群体进行协作，每个子群体都是广义的多层感知器。与标准的多层感知器相比，基于GMLP的网络结构具有相对较少的节点和连接数，可以使用更少的节点定义非常复杂的表面。同时，较小的网络进行演进能够提高网络的可解释性。【 GARCÍA- PEDRAJAS N，ORTIZ- BOYER D，HERVAS MARTINEZ C.Cooperative coevolution of generalized multi-layer perceptrons[J].Neurocomputing，2004，56：257-283】

- 受到大脑中神经胶质特征的启发，Ikuta 等人提出一种具有神经胶质网络的多层感知器，其中神经胶质网络仅与 MLP 的第二层隐藏层进行连接，通过计算机仿真结果证实具有神经胶质网络的 MLP 相对于标准的 MLP 具有更好的性能，赋予了 MLP中神经元的位置依赖性Li等提出一种基于简化几何代数（Reduced Geometric Algebra，RGA）的多层感知器扩展模型RGA-MLP，传统MLP模型将每个维度的信号视为一个实数进行单独处理，基于 RGA 的模型中输入、输出、激活函数以及运算符都使用可交换乘法规则扩展到 RGA域，并且使用 RGA版本的反向传播训练神经网络，用于多维信号处理，将多个通道视为一个单元而不是一个单独的组件，可以实现更高的分类精度、更快的收敛速度以及更低的计算复杂度。【LIYP,CAOWM. An extended multilayer perceptron model using reduced geometric algebra[J].IEEE Access，2019.7:129815-129823】

- Masulli和Penna 将基于主成分分析的增量输入维度（IID）算法应用于MLP中，提高了多层感知器的学习速率。【MASULLI F，PENNA M.Improving learning speed in multilayer perceptrons through principal component analysis[J].Proceedings of the SPIE，1996，2760：85-95】

- Martinez-Morales等人提出通过多目标蚁群优化算法对 MLP 参数进行优化的 MLP-MOACO 模型，对发动机污染物相关系数进行计算以及估算发动机的废气排放。【MARTINEZ-MORALES J，QUEJ-COSGAYA H，LAGUNAS JIMENEZ J，et al.Design optimization of multilayer perceptron neural network by ant colony optimization applied to engine emissions data[J].Science China-Technological Sciences，2019，62（6）：1055-1064】 

- Mosavi 等 人 提 出 MLP- GWO 模 型 ，该 模 型 将Gray Wolf 算法与标准 MLP 模型结合在一起并应用于土壤电导率预测，实验结果证明混合MLP-GWO模型相对于标准MLP模型可以在隐藏层获取更加准确的连接权重，从而提高预测精度[16]。Liu等基于 Adaboost（自适应 Boosting）算法和 MLP（多层感知器）神经网络，提出了四种不同的混合方法用于高精度多步风速预测，证明了Adaboost算法能有效提高MLP神经网络的性能。【 LIU H，TIAN H Q，LI Y F，et al.Comparison of four Adaboost algorithm based artificial neural networks in wind speed predictions[J].Energy Conversion and Management，2015，92：67-81】

### 2.2 反向传播神经网络

#### 2.2.1 **BP**神经网络

BP神经网络模型（反向传播算法）的网络体系结构是多层的，本质上是一种梯度下降局部优化技术，与网络权重的向后误差校正相关。

![BP神经网络结构]({{ "/assets/images/neural-network-review/image-20250306104600125.png" | relative_url }})

**工作原理：**

BP神经网络是一种按误差反向传播(简称误差反传)训练的多层前馈网络，其算法称为BP算法，它的基本思想是梯度下降法，利用梯度搜索技术，以期使网络的实际输出值和期望输出值的误差均方差为最小。

- **BP神经网络核心步骤：正向传播与反向传播**

  （实线代表正向传播，虚线代表反向传播）

![BP神经网络传播过程]({{ "/assets/images/neural-network-review/image-20250306110320152.png" | relative_url }})

- 梯度下降法

![梯度下降法]({{ "/assets/images/neural-network-review/gradient-descent.png" | relative_url }})

**优缺点：**

BP 神经网络的多层结构使得模型的输出更加准确，但BP神经网络仍然存在一定的缺陷。针对XOR之类的非线性可分问题时，使用 BP 神经网络可能出现局部最小值导致无法找到全局最优解，并且在面对大样本数据时均方误差MSE过大导致难以收敛。

**应用案例：**

1)函数逼近：用输入向量和相应的输出向量训练一个网络逼近一个函数。
2)模式识别：用一个待定的输出向量将它与输入向量联系起来。
3)分类：把输入向量所定义的合适方式进行分类。
4)数据压缩：减少输出向量维数以便于传输或存储。

**改进算法：**

- 王丽红等人将传统 BP 组合起来构成 AdaBoost-BP 模型[18]。

![AdaBoost-BP模型]({{ "/assets/images/neural-network-review/image-20250306103124724.png" | relative_url }})

【王丽红.基于 BP-AdaBoost的电商短期销量预测模型[J].计算机系统应用，2021，30（2）：260-264】

- 针对BP网络使用梯度下降容易使模型陷入局部最优的缺陷，黄宝洲等人改变传统 BP 调整自身阈值和权重参数的方式，使用粒子群优化算法获取 BP 网络的权重和阈值参数。【黄宝洲，杨俊华，卢思灵，等.基于改进粒子群优化神经网络算法的波浪捕获功率预测[J].太阳能学报，2021，42（2）：302-308.】

### 2.3 卷积神经网络（CNN）

#### **2.3.1 LeNet-5**

Lenet 是一系列网络的合称，包括 Lenet1 - Lenet5，由 Yann LeCun 等人 在1990年《Handwritten Digit Recognition with a Back-Propagation Network》中提出，是卷积神经网络的开山之作，也是将深度学习推向繁荣的一座里程碑

![LeNet-5结构]({{ "/assets/images/neural-network-review/image-20250306104838181.png" | relative_url }})

**工作原理：**

LeNet-5 网络结构并不是全连接网络，LeCun 等人使用多个卷积核，采用卷积核权值共享的方法减少卷积神经网络中的连接数，模型更加简洁易于计算。其网络体系由七层结构组成。

- 卷积层

![卷积层原理]({{ "/assets/images/neural-network-review/image-20250306111420256.png" | relative_url }})

- 池化层（最大池化层Max pooling）

![池化层原理]({{ "/assets/images/neural-network-review/image-20250306111404242.png" | relative_url }})

**优缺点：**

LeNet-5 的效率较低，单通道的网络结构进行特征提取时不完整并且模型收敛速率慢

**改进算法：**

- 安源等人采用四通道网络，对四个通道的卷积核和偏置参数进行设置，采用 ReLU 激活函数，在 MNIST 数据集上四通道模型准确率为96.56%，比传统LeNet-5高出4.52%，但这种多通道结构目前不能对数据规模进行动态调整。
- Hou等人提出使用FPGA加速LeNet-5来改进原始的LeNet-5模型，最后通过训练手写数字识别模型实验，证实了改进后的模型在效率和准确度上都有很大的提升。
- 针对滚动轴承故障诊断中传统LeNet-5网络识别准确率较低，模型收敛速率慢，泛化能力弱的问题，Wan等人提出了一种采用改进的二维LeNet-5网络的滚动轴承故障诊断方法，调整卷积核数量以及大小并执行批归一化，全连接层除最后一层外均进行删除操作，增强网络泛化能力，实验表明改进方法具有更高的故障诊断精度和更少的训练时长。

#### 2.**3.2 AlexNet**

AlexNet是由Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton在2012年ImageNet图像分类竞赛中提出的一种经典的卷积神经网络。当时，AlexNet在 ImageNet 大规模视觉识别竞赛中取得了优异的成绩，把深度学习模型在比赛中的正确率提升到一个前所未有的高度。因此，它的出现对深度学习发展具有里程碑式的意义。

![AlexNet结构]({{ "/assets/images/neural-network-review/alexnet-structure.jpg" | relative_url }})

**工作原理：**

- AlexNet在LetNet的基础上更进一步加深了网络结构，是一个 5+3 的卷积神经网络，包括 5 个卷积层、3 个全连接层。

- 首次使用了修正线性单元（ReLU）这一非线性激活函数。相比于传统的 sigmoid 和 tanh 函数，ReLU 能够在保持计算速度的同时，有效地解决了梯度消失问题，从而使得训练更加高效。

- 为了防止过拟合，AlexNet 引入了数据增强和 Dropout 技术。

  ![ReLU激活函数]({{ "/assets/images/neural-network-review/relu-activation.jpg" | relative_url }})

**优缺点：**

AlexNet 相比于传统 CNN 在图像领域具有更高的识别率、图像质量提高，但由于其对特征进行提取时使用的卷积核不具有多样性，在进行图像识别时仍然存在一定的误差。

**改进算法：**

- 郭书杰等人在使用AlexNet对手势识别时提出AlexNet的非线性激活函数会导致训练过程中出现神经元死亡，因此设计了包含三个批归一化的AlexNet 结构（针对 3、4、5 层做批归一化）并且优化了模型超参数，结构优化后的 AlexNet 准确率提高了约4%，但该模型和原始模型同样受限于输入图像的复杂性和手势在图像面积中的占比。
- 黄方亮等人提出了AlexNet_En模型，该模型在原始 AlexNet的第四层后添加了一层与第四层相同的卷积操作，采用 384个 3×3卷积核确实增加了模型的准确率，在 ImageNet 数据集上达到 94.00%，但该结构同样使模型复杂度变高，计算需求增加带来了一定的硬件负担。
- AlexNet在场景分类应用中，传统AlexNet卷积核跨度大导致特征图的分辨率下降过快，Xiao等人提出一种改进的 AlexNet模型，将大卷积核分解为两个步幅较小的小卷积核级联结构，实验证明改进模型在 23 种场景分类中的分类精度高于原始的 AlexNet模型。Han等人提出一种改进的预训练 AlexNet 体系结构 AlexNetSPP-SS，结合了比例池-空间金字塔池（SPP）和边监督（SS）来改善原始 AlexNet不收敛以及过拟合的问题，并证明了经过预训练的 AlexNet-SPP-SS 模型优于原始的AlexNet体系结构以及传统的场景分类方法。

#### 2.**3.3 GoogLeNet**

GoogLeNet在2014年由Google团队提出， 斩获当年ImageNet(ILSVRC14)竞赛中Classification Task (分类任务) 第一名，VGG获得了第二名，为了向"LeNet"致敬，因此取名为"GoogLeNet"

![GoogLeNet结构]({{ "/assets/images/neural-network-review/googlenet-structure.png" | relative_url }})

**工作原理：**	

GoogLeNet 总共有22层，由 9 个 Inception v1 模块和 5 个池化层以及其他一些卷积层和全连接层构成。该网络有3个输出层，其中的两个是辅助分类层。

- inception v1结构

  为了既保持网络结构的稀疏性，又能利用密集矩阵的高计算性能，GoogLeNet提出了一种并联结构，Inception网络结构。其主要思想是寻找用密集成分来近似最优局部稀疏连接，通过构造一种"基础神经元"结构，来搭建一个稀疏性、高计算性能的网络结构。

<img src="{{ '/assets/images/neural-network-review/image-20250306112237329.png' | relative_url }}" alt="inception结构" style="zoom:67%;" />

- 辅助分类器（Auxiliary Classifier）

  网络主干右边的两个分支就是辅助分类器，他们也能预测图片的类别，其结构一模一样。它确保了即便是隐藏单元和中间层也参与了特征计算，在inception网络中起到一种调整的效果，避免梯度消失。

<img src="{{ '/assets/images/neural-network-review/image-20250306112254129.png' | relative_url }}" alt="辅助分类器" style="zoom:67%;" />

**优缺点：**

GoogLeNet相对于AlexNet等网络小、参数较少，性能相对优越，GoogLeNet使用Inception网络结构，保持神经网络的稀疏性并且提高了性能。复杂性较高，针对小规模数据时可能无法达到大规模数据集所能达到的性能。

**改进算法：**

- Zhu等人提出一种新颖的双重微调策略来训练GoogLeNet模型，通过截断操作优化GoogLeNet的结构减小网络大小，用于极端天气识别，在天气数据集上进一步微调得到最后的型，优化后的模型大小为原始 GoogLeNet的 31.23%，但识别准确率从 94.74%提升至 95.46%，识别速度也有所提高。
- Tuan使用预训练的三个神经网络AlexNet、GoogLeNet和SqueezeNet，并对神经网络进行了微调，用于COVID-19、病毒性肺炎和正常胸部 X 射线图像的分类，从不同性能指标的训练和测试数据中证实模型的有效性。使用传统的 GoogLeNet 深层次网络结构做特征提取时可能会因为感受野扩大而导致特征消失，对准确率产生影响。要提升传统模型的性能，加大深层网络的深度和宽度会很大程度增加参数量，产生更大的计算负担，一般对传统模型进行结构优化。
- 传统GoogLeNet包 含 9个 Inception模块，张泽中等人在对胃癌病理图像提取特征时以 Inception模块为单位进行实验，发现在第 7个模块模型取得最优性能，最终保留前 7 个 Inception，GPU占用由传统的 65%降为 43%，训练时间少了约 4小 时，第 7 个模块后衔接全卷积网络对特征分类输出，30次迭代后模型准确率为 99.28%，但模型在提高灵敏度的前提下损失了部分特异度。

#### 2.**3.4 **ResNet**

ResNet（Residual Neural Network）由微软研究院的Kaiming He等四名华人提出，通过使用ResNet Unit成功训练出了152层的神经网络，并在ILSVRC2015比赛中取得冠军，在top5上的错误率为3.57%，同时参数量比VGGNet低，效果非常突出。

![ResNet结构]({{ "/assets/images/neural-network-review/resnet-structure.png" | relative_url }})

**工作原理：**	

- 提出residual结构（残差结构），并搭建超深的网络结构(突破1000层)

![残差结构]({{ "/assets/images/neural-network-review/image-20250220112258644.png" | relative_url }})

- 使用Batch Normalization加速训练(丢弃dropout)

  ![Batch Normalization]({{ "/assets/images/neural-network-review/image-20250306113145769.png" | relative_url }})

**优缺点：**

残差神经网络（ResNet）在多图像处理问题中能获取高精度的输出结果。其主要功能称为跳跃连接，有助于梯度流动。残差神经网络中 He 等人利用多层的神经网络结构来拟合残差映射的效果，从而解决加深神经网络深度导致的梯度消失以及精度下降等问题。

![残差网络原理]({{ "/assets/images/neural-network-review/image-20250306113125547.png" | relative_url }})	经典 ResNet 还存在很大的改进空间，残差单元中通过最终梯度所包含的梯度信息无法直接计算出其他梯度信息，导致残差单元增多时出现更多的卷积层无法获得梯度信息传递。

**改进算法：**

- 李国强等人提出 FCM-Resnet，提出跨层连接将所有卷积层与平均池化层相连，使每个残差单元都能传递梯度信息，在对比 FCM-Resnet- 56、FCM-Resnet-110 和传统 Resnet 实验结果后提出的改进模型准确率为 99.57%和 99.63%，上升了约 0.03%和 0.02%，改进模型的稳定性和优化还存在改进空间。
- 使用 1×1 卷积核来解决 ResNet50 输入输出数据维度不匹配时，在细颗粒图像分类领域会丢失信息并且影响计算结果，李晓双等人把跳跃连接中步长为2的卷积核替换为步长 1，并在卷积操作前加入了平均池化，一定程度上保留了梯度信息，仅在小样本下证明了模型优化有效。
- Wu 等人利用残差网络（ResNet）、双向门控单元（BiGRU）和注意力机制提出一种基于神经网络和主动学习（DABot）的新浪微博社交机器人检测框架，经过性能评估后，DABot的精度为 0.988 7，说明该模型更加有效。

#### 2.3.5 CNN代表性模型对比

| 模型名称  | 发表年份 | 机制                                 | 优点                                                 | 缺点                               | 应用案例                             |
| --------- | -------- | ------------------------------------ | ---------------------------------------------------- | ---------------------------------- | ------------------------------------ |
| LeNet     | 1998     | 卷积核权值共享                       | 结构简单，易于实现                                   | 适用范围有限，处理复杂图像能力不足 | MNIST手写数字识别                    |
| AlexNet   | 2012     | 深层CNN，使用ReLU激活和Dropout正则化 | 显著提升图像分类准确率，推动深度学习发展             | 结构较为庞大，计算资源需求高       | ImageNet图像分类                     |
| VGG       | 2014     | 使用大量3x3卷积核，网络深度较大      | 提升模型性能，结构统一，易于迁移学习                 | 参数量巨大，计算和存储开销高       | 图像分类、目标检测                   |
| GoogLeNet | 2014     | Inception单元模块                    | 稀疏性，模块化，更好地提取                           | 占用资源大                         | 针对微观复杂图像处理效果较好         |
| ResNet    | 2015     | 引入残差连接，允许训练更深层网络     | 上下文信息解决深层网络的退化问题，提升性能，训练稳定 | 结构复杂，计算资源需求高           | ImageNet图像分类、目标检测、语义分割 |

### 2.4 递归神经网络（RNN）

#### 2.4.1 标准RNN（Elman）

RNN用于处理序列数据。在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的。但是这种普通的神经网络对于很多问题却无能无力。例如，你要预测句子的下一个单词是什么，一般需要用到前面的单词，因为一个句子中前后单词并不是独立的。RNN之所以称为循环神经网路，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。理论上，RNN能够对任何长度的序列数据进行处理。但是在实践中，为了降低复杂性往往假设当前的状态只与前面的几个状态相关。

![标准RNN结构]({{ "/assets/images/neural-network-review/image-20250306120556829.png" | relative_url }})

**工作原理：**

**循环神经网络**的**隐藏层**的值s不仅仅取决于当前这次的输入x，还取决于上一次**隐藏层**的值s。**权重矩阵** W就是**隐藏层**上一次的值作为这一次的输入的权重。

![RNN工作原理]({{ "/assets/images/neural-network-review/rnn-working-principle.jpg" | relative_url }})

按时间线将循环神经网络展开，这个网络在t时刻接收到输入 x~t 之后，隐藏层的值是st ，输出值是 ot 。关键一点是， st 的值不仅仅取决于 xt ，还取决于 st−1 。我们可以用下面的公式来表示**循环神经网络**的计算方法：

<img src="{{ '/assets/images/neural-network-review/rnn-formula.jpg' | relative_url }}" alt="RNN公式" style="zoom: 33%;" />

表达得更直观的图有：

![RNN展开结构]({{ "/assets/images/neural-network-review/rnn-unfolded.png" | relative_url }})

![RNN详细结构]({{ "/assets/images/neural-network-review/rnn-detailed.png" | relative_url }})

**优缺点：**

递归神经网络RNN的隐藏层结构使其在时间序列预测方面具有非常广泛的应用。传统RNN 会产生梯度消失，并且在处理数据长期依赖时精度会大幅度下降，输入输出数据序列不匹配，模型的参数共享引起的缺失信息可能对时序特征产生影响，RNN 将每个节点的先前隐藏状态进行编码作为整个模型的历史信息，但是忽略了每个节点之间的独立关系。

**应用案例：**

- Wei等人将MLP、RNN、LSTM、GRU 分别应用于孔隙水压力（PWP），得出具有RNN结构的模型在针对时间序列数据时相较于MLP更为准确
- Ling等人将 RNN 应用于核动力机械的故障预测，提出一种智能故障预测方法，将主成分分析 PCA 降维后的数据传递给完整的 RNN 模型，根据转速和振动信号分别提前 60 h 和 44 h 生成警报。实验结果表明，RNN 模型可以有效地识别蠕变期间的故障

**算法改进：**

- Stender等人将CNN与RNN结合使用，用于刹车噪声检测和预测，发现结合模型可以克服传统方法的局限性，第一部分采用 CNN 显示出了优越的检测质量和特征提取性能，第二部分采用的 RNN 依赖于噪声的瞬时频谱特性，使用该模型预测刹车噪音的精度和准确度都非常高，该模型在声音检测方面展现出巨大的潜力。

#### 2.4.2 长短期递归记忆神经网络LSTM

1997年Hochreiter和Schmidhuber在标准RNN中引入门控单元概念，解决了标准 RNN 存在的梯度消失问题

![LSTM结构]({{ "/assets/images/neural-network-review/image-20250306120612591.png" | relative_url }})

**工作原理：**

所有 RNN 都具有一种重复神经网络模块的链式的形式。在标准的RNN中，这个重复的模块只有一个非常简单的结构，例如一个 tanh层。 

                     

标准 RNN 中的重复模块包含单一的层。

LSTM 同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于 单一神经网络层，这里是有四个，以一种非常特殊的方式进行交互。

<img src="{{ '/assets/images/neural-network-review/lstm-structure.png' | relative_url }}" alt="LSTM结构" style="zoom:80%;" />                         

LSTM 中的重复模块包含四个交互的层。

<img src="{{ '/assets/images/neural-network-review/lstm-detailed.png' | relative_url }}" alt="LSTM详细结构" style="zoom: 80%;" />                          

在上面的图例中，每一条黑线传输着一整个向量，从一个节点的输出到其他节点的输入。粉色的圈代表 pointwise 的操作，诸如向量的和，而黄色的矩阵就是学习到的神经网络层。合在一起的线表示向量的连接，分开的线表示内容被复制，然后分发到不同的位置。

![LSTM细胞状态]({{ "/assets/images/neural-network-review/lstm-cell-state1.png" | relative_url }})

LSTM 的关键就是细胞状态（cell），水平线在图上方贯穿运行。细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。 

![LSTM门控机制]({{ "/assets/images/neural-network-review/lstm-gates.png" | relative_url }})

LSTM 有通过精心设计的称作为"门"的结构来去除或者增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作。 

<img src="{{ '/assets/images/neural-network-review/lstm-sigmoid.png' | relative_url }}" alt="LSTM Sigmoid层" style="zoom:67%;" />

Sigmoid 层输出 0 到 1 之间的数值，描述每个部分有多少量可以通过。0 代表"不许任何量通过"，1 就指"允许任意量通过"！

LSTM中有3个控制门：输入门，输出门，记忆门。

（1）forget gate：选择忘记过去某些信息：

<img src="{{ '/assets/images/neural-network-review/lstm-forget-gate.gif' | relative_url }}" alt="遗忘门" style="zoom:67%;" />

（2）input gate：记忆现在的某些信息：

<img src="{{ '/assets/images/neural-network-review/lstm-input-gate.gif' | relative_url }}" alt="输入门" style="zoom:67%;" />

（3） 将过去与现在的记忆进行合并：

<img src="{{ '/assets/images/neural-network-review/lstm-merge.gif' | relative_url }}" alt="记忆合并" style="zoom:67%;" />

（4）output gate：输出

<img src="{{ '/assets/images/neural-network-review/lstm-output-gate.gif' | relative_url }}" alt="输出门" style="zoom:67%;" />

公式总结：                                  
<img src="{{ '/assets/images/neural-network-review/lstm-formula.png' | relative_url }}" alt="LSTM公式总结" style="zoom:67%;" />

                                                              

**应用案例：**                                                 

- Wang 等人将LSTM 应用于语音增强，提出一种 LSTM-卷积-BLSTM编解码器网络（LCLED），包含了转置卷积和跳跃连接，使用两个LSTM单元对上下文信息进行捕获，使用卷积层对频域特征进行提取，在多种噪音的情况下该网络模型仍具有良好的降噪功能，在语音增强方面具有更高的鲁棒性。【WANG Z Y，ZHANG T，SHAO Y Y，et al.LSTM convolutional- BLSTM encoder- decoder network for minimum mean-square error approach to speech enhance ment[J].Applied Acoustics，2021，172：1-7.】
- Petmezas 等人将LSTM与CNN结合提出CNN-LSTM模型应用于手动心电图（ECG）中，通过 CNN 将提取到的 ECG 信号特征传递给LSTM以实现时间动态记忆，从而更为准确地分类四种 ECG 类型。最终使用该模型在 MIT-BIH 心房颤动数据上进行训练，采用十折交叉验证了该模型能准确验证ECG类型（灵敏度为97.87%，特异性为99.29%），可以帮助临床医生实时检测常见类型的房颤。【 PETMEZASG，HARISK，TEFANOPOULOS L，et al. Automated atrial fibrillation detection using a hybrid CNN- LSTM network on imbalanced ECG datasets[J]. Biomedical Signal Processing and Control，2021，63：1-9.】
- 赵红蕊等人将LSTM与CNN结合用于股票价格预测并引入注意力机制（Convolutional Block Attention Module，CBAM），提出一种 LSTM-CNN-CBAM 混合模型，对比实验结果验证了在LSTM-CNN结合模型中加入CBAM模块的可行性。

#### 2.4.3 RNN代表性模型对比

| 模型名称         | 发表年份 | 主要特点                                   | 优点                                 | 缺点                                 | 应用案例                       |
| ---------------- | -------- | ------------------------------------------ | ------------------------------------ | ------------------------------------ | ------------------------------ |
| 标准RNN（Elman） | 1990     | 基本的循环结构，能够处理序列数据           | 结构简单，适用于基本的序列任务       | 难以捕捉长距离依赖，易梯度消失或爆炸 | 基本时间序列预测、简单文本生成 |
| LSTM             | 1997     | 引入记忆单元和门控机制，解决长距离依赖问题 | 能有效捕捉长期依赖，缓解梯度消失问题 | 结构复杂，计算量较大                 | 机器翻译、语音识别、文本生成   |
| GRU              | 2014     | 简化的LSTM结构，仅使用更新门和重置门       | 计算效率高，参数较少                 | 在某些任务上性能略逊于LSTM           | 实时语音识别、移动设备应用     |

## 三、总结与展望

本文对人工神经网络的发展历程进行了系统梳理，重点介绍了多层感知器、反向传播神经网络、卷积神经网络和递归神经网络等主流模型的基本原理、优缺点及应用场景。

随着计算能力的提升和大数据技术的发展，神经网络在各个领域展现出强大的应用潜力。未来神经网络的发展趋势可能包括：

1. **模型轻量化**：在保持性能的同时减少模型参数和计算复杂度
2. **跨模态学习**：整合视觉、语言、听觉等多种模态信息
3. **可解释性增强**：提高模型决策过程的透明度和可理解性
4. **自监督学习**：减少对标注数据的依赖，提高学习效率

神经网络技术的持续创新将为人工智能的发展注入新的活力，推动更多实际应用场景的落地。

---

*本文基于学术文献和技术资料整理，旨在为读者提供神经网络领域的系统性概述。如有不准确之处，欢迎指正。*