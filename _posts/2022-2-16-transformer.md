---
layout: post
title: 'Transformer模型'
date: 2022-2-16
author: 不显电性
cover: 'http://commcheck396.github.io/assets/img/2022_2_14/transformer.png'
tags: ML Python
---
> Attention is all U need


在了解transformer模型之前，我们首先要搞清self-attention的概念。 
Self-attention，输入是一串vector set，输出亦然，RNN网络同样可以实现类似的事情而且更好搭建，但是Self-attention可以实现数据的并行处理，而RNN仅可以实现串行，所以优先研究这个效率较高的方向了，也可能会去学一下RNN，~~因为这个搭建起来实在是太麻烦了~~，放在Pytorch便签中吧。   

Self-attention其实不难理解，简而言之就是用各种方法在输入的向量间找彼此的关系α，然后对输入内容进行预测，输出一个vector set。直接上图。
![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/selfattention.jpg)
这是self-attention的整个流程，并非神经网络！若要进行机器学习训练，还需要搭建神经网络，这也便有了transformer模型。  
其实，transformer模型和上述过程并非完全相关，与之更为相关的是下方的multihead
![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/multi.jpg)
看过了整个路程，不难发现我们需要学习的参数一共就下面几个儿
![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/parameter.jpg)
![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/transformer.gif)