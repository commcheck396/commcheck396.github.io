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
在原始论文中Self-Attention中没有考虑位置信息，不妨加一个ei来表示位置信息，怎么理解呢，可以理解为在xi向量上加了一个one-hot表示的pi，然后经过计算发现ei并不影响原来的向量，所以加入这个位置信息不仅不会影响已有的数据，还能在输入中加入有关位置的信息，可谓一举两得。
![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/position.png)

其实，transformer模型和上述过程并非完全相关，与之更为相关的是下方的multihead
![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/multi.jpg)
看过了整个路程，不难发现我们需要学习的参数一共就下面几个儿
![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/parameter.jpg)
Self-attention也就这么多，下面进入正题transformer。

## Transformer实现
这个模型可以看成是一个黑箱操作。在机器翻译中，就是输入一种语言，输出另一种语言。
![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/transformer.png)
这个黑箱是由编码组件、解码组件和它们之间的连接组成。

![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/blackbox.png)

编码组件部分由一堆编码器（encoder）构成（论文中是将6个编码器叠在一起）。解码组件部分也是由相同数量（与编码器对应）的解码器（decoder）组成的。所有的编码器在结构上都是相同的，但它们没有共享参数。每个解码器都可以分解成两个子层。

![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/bianmaqi.png)

从编码器输入的句子首先会经过一个上文提到的自注意力（self-attention）层，这层帮助编码器在对每个单词编码时关注输入句子的其他单词。  

自注意力层的输出会传递到前馈（feed-forward）神经网络中。每个位置的单词对应的前馈神经网络都完全一样。  

解码器中也有编码器的自注意力（self-attention）层和前馈（feed-forward）层。除此之外，这两个层之间还有一个注意力层，用来关注输入句子的相关部分（和seq2seq模型的注意力作用相似）。  

Transformer 的 Decoder的输入与Encoder的输出处理方法步骤是一样地，一个接受source数据，一个接受target数据，举个例子：Encoder接受英文"Tom chase Jerry"，Decoder接受中文"汤姆追逐杰瑞"。只是在有target数据时也就是在进行有监督训练时才会接受Outputs Embedding，进行预测时则不会接收。  


之后就要引入我们的张量了，我们首先将每个输入单词通过词嵌入算法转换为词向量，每个单词都被嵌入为512维的向量。

词嵌入过程只发生在最底层的编码器中。所有的编码器都有一个相同的特点，即它们接收一个向量列表，列表中的每个向量大小为512维。在底层（最开始）编码器中它就是词向量，但是在其他编码器中，它就是下一层编码器的输出（也是一个向量列表）。向量列表大小是我们可以设置的超参数——一般是我们训练集中最长句子的长度。  

我们还需要给每个word的词向量添加位置编码positional encoding，为什么需要添加位置编码呢？  

首先咱们知道，一句话中同一个词，如果词语出现位置不同，意思可能发生翻天覆地的变化，就比如：我欠他100W 和 他欠我100W。这两句话的意思一个地狱一个天堂。可见获取词语出现在句子中的位置信息是一件很重要的事情。  

这positional encoding的获取也是一门学问，一般我们会用下面两个公式来获取。  

啥？你问为啥？别问，问就是古圣先贤。

![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/positionf.png)

### encoder

self-attention结构如下图所示：

![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/encodelayer.png)

但在encoder layer中运用的架构并非这一个，而是Multi-Head Attention，这个问题在上文也有讨论过，其实它就是在self-attention的基础上，对于输入的embedding矩阵有多个矩阵进行数据的处理，并在得到多个结果后再进行降维，得到最终结果。  

而这个降维操作，展开来说就是Add＆Normalize  










![pic from internet](http://commcheck396.github.io/assets/img/2022_2_14/transformer.gif)