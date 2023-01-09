# Classification for Chinese Text

## 1. Transformer原理

### 1. 模型结构

​		Transformer是一个**Encoder-Decoder**结构的神经网络结构。Encoder将输入序列$(x_1, x_2,...x_n)$映射到一个连续表示序列$z=(z_1, z_2,...z_n)$，而对于编码得到的z，Decoder每次解码生成一个符号，直到生成完整的输出序列： $(y_1, y_2,...y_n)$ 。每一步解码的过程中，模型都是自回归的——即在生成下一个符号时将先前生成的符号作为附加输入。

​		下图是Transformer的示意图，左右两部分分别是Encoder和Decoder，$N\times$表示有N个相同的层(layer)，在原文《Attention is all you need》中，设定了N的值为6 。

​		在Encoder中，每一个layer分别由两层组成：第一层是一个多头自注意机制层（multi-head self-attention mechanism），第二层是一个简单的positionwise fully connected前馈网络，在两个子层之间使用残差连接(residual connection)，之后再进行层归一化(layer normalization)。

​		Decoder的结构与Encoder相似，除了Encoder中的两个子层之外，Decoder多插入了一个子层，用来对Encoder的输出执行multi-head self-attention mechanism。

<img src="https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221107153305594.png" height=500>



### 2. Attention

​		Attention函数可以将Query和一组Key-Value对映射到输出，其中Query、Key、Value和输出都是向量。 输出是值的加权和，其中分配给每个Value的权重由Query与相应Key的兼容函数(compatibility function)计算。

#### (1) Scaled Dot-Product Attention

​		Transformer中对Attention机制的应用称为“Scaled Dot-Product Attention”，原理如图所示，输入包含$d_k$维的Query和Key，以及$d_v$维的Value。 首先分别计算Query与各个Key的点积，然后将每个点积除以$d_k$，最后使用Softmax函数来获得Key的权重。

​		在具体实践中，将所有的Query、Key和Value向量分别组合成矩阵Q、K和V，从而能够加速运算，这样输出矩阵可以表示为：
$$
Attention(Q, K, V ) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$


#### (2) Multi-Head Attention

​		《Attention is All you need》中提出，将$d_{model}$维的query、key和value分别投影到$d_k, d_k,d_v$维k次，能够获得比对其执行单次的Attention效果要好。在投影之后，并行地执行Attention，生成一个$d_v$维的输出值，将这些输出值连接起来再次进行投影，产生最终值，如图所示。

​		具体实践中 Multi-Head Attention的计算方式如下：
$$
MultiHead(Q,K,V)=Concat(head_1,...head_h)W^O \\
其中head_i=Attention(QW_i^Q,KW_i^K,VW_i^V),\quad W_i^{Q、K、V}\in \mathbb{R}^{d_{model} \times d_{k、k、v}}
$$
​		原文中有如下的设置：$h=8,\quad d_k=d_v=d_{model}/h=64$ 。



### 3. Transformer用于文本分类

​		在文本分类任务中运用Transformer模型，只需借助Encoder部分，在Encoder层之上再加入一层全连接层，即能执行分类任务，模型结构如下图所示(模型的可视化结果)：

<img src="https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221121162809649.png" height=500>



## 2. BiLSTM

### (1)模型结构

​		LSTM全称为长短期记忆(Long Short-Term Memory)，是一种时间循环神经网络（RNN）。BiLSTM则是由一个前向的LSTM与一个反向的LSTM连接而成。RNN与BiLSTM的基本结构如下图所示。

<table><tr>
<td><img src="https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221121164341840.png" width=400 border=0></td>
<td><img src="https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221121164422304.png" width=400 border=0></td>
</tr></table>




### (2) BiLSTM用于文本分类任务

​		输入向量在经过BiLSTM之后再接入一层Attention，和两层的全连接层得到输出，模型的结构如下图所示(同Transformer)：

<img src="https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221121165953745.png" height=400>





## 3. 使用预训练的词向量执行上述任务

​		预训练的词向量中共包含有635974个词，每个词的维度为 300维。使用预训练的词向量的一般思路为：先根据预训练的embedding的均值和标准差随机初始化当前数据集的词向量，然后再依次判断预训练的词向量中是否有当前数据集的词，若有则用预训练词向量做替换，没否则保持随机初始化的词向量。(参考自网络博客:[地址]([如何使用预训练的word embedding - CodeAntenna](https://codeantenna.com/a/h8ycaloGRJ)))。

​		在本任务中，处理的方式为：在数据处理器`Exp_Dataset.py`内加入如下的代码段和处理函数：

```python
# 若要采用预训练的 embedding, 需在此处对 sent 进行分词
if self.pre_train:
    seg_sent = jieba.cut(sent)
    for words in seg_sent:
        if words in word_dic:
            pass
        else:
            word_dic[word] = one_word_index
            one_word_index += 1
 ...
if self.pre_train:
    self.weight_matrix = load_vector(word_index=word_dic)
```

​		运行程序时给出是否需要加载预训练权重的命令，在模型构建时传入相应的参数，代码运行命令见于文末，处理函数详见处理器中的`load_vector(word_index)`。



## 4. 实验结果

### Transformer

<img src="https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221121172234130.png" height=400>

### BiLSTM

<img src="https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221121172331323.png" height=400>



### 预训练的Transformer

<img src="https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221121172405437.png" height=400>

### 预训练的BiLSTM

![image-20221121172756761](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221121172756761.png)







## Appendix：程序命令行参数



```
--model 选择Transformer_model 或 BiLSTM_model
--embedding  选择随机初始化或选择预训练的词向量
```











