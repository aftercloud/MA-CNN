# MA-CNN
Multi-Attention-CNN
## 说明 
+ 本仓库只是对ICCV 2017 论文《Learning Multi-Attention Convolutional Neural Network for Fine-Grained Image Recognition》中的多注意力卷积神经网络进行模仿和复现  
+ 论文作者GitHub地址 https://github.com/Jianlong-Fu/Multi-Attention-CNN  
+ 本项目的重点是全连接层聚类获得的**注意力矩阵**和**通道损失函数**  
+ 本项目主要用于图像中**细粒度特征**的识别和视频流中**关键帧**的提取  
+ 本项目由**Pytorch**实现，只能运行在**GPU**上
## 代码
+ data_macnn.py  
对外提供数据接口，面对不同数据集需要修改继承于*Pytorch*数据相关模块的三个函数  
+ conv_macnn.py  
训练*macnn*的卷积部分，可以任选卷积网络，注意好输出的通道数量和尺寸就好
+ part_macnn.py  
训练*macnn*的聚类部分  
>Dis已实现，Div损失只提供示例代码（注释区域），按需修改  

>代码中的注意力区域为1个，若需要为多个，参照现有代码扩增，同时对于损失函数和分类器在维度上也要扩增  
+ clf_macnn.py  
训练*macnn*的分类部分，此部分可以不要，也可以直接修改成*softmax*，还可以任意使用深层神经网络进行分类
## 原理  
![macnn](https://img-blog.csdn.net/20180511154450998?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VsbGluX3lvdW5n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
>原始输入矩阵包含若干通道，经过卷积后可以获得更多通道  
>每个通道响应原始输入矩阵中的特定特征  
>对通道进行聚类，聚簇质心所在位置即为细粒度特征  

>使用*n*组全连接层用来确定*n*个细粒度特征  
>对于每组全连接层,每个通道要么属于该组全连接层，要么不属于该组全连接层(所代表的细粒度特征)  
>(理论上来说)全连接层对通道进行二分类

>训练过程中，全连接层会对输入的通道数据的每一个二维层次的坐标产生一个权重  
>将权重和输入通道数据在对应位置点乘得到注意力矩阵  
