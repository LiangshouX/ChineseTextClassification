import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


def clone(module, N):
    """生成N个相同的layer"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'Transformer'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(dim_model, num_head, dropout)
        self.feed_forward = PositionWiseFeedForward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class ScaledDotProductAttention(nn.Module):
    """原文提出的Attention，d_k维度的Q和K，d_v维的V"""

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    @staticmethod
    def attention(Query, Key, Value, scale=None, mask=None):
        """Attention的计算
        Args:
            Query(Tensor): [batch_size, len_Q, dim_Q],also [batch_size, len_Q, dim_K]
            Key(Tensor):  [batch_size, len_K, dim_K]
            Value(Tensor): [batch_size, len_V, dim_V]
            scale(float): scaling factor given by the paper, sqrt(dim_K)
            mask(bool):  optional
        Returns:

        """
        dim_k = Key.shape[-1]
        if not scale:
            attention_ = torch.matmul(Query, Key.permute(0, 2, 1)) / np.sqrt(dim_k)
        else:
            attention_ = torch.matmul(Query, Key.permute(0, 2, 1)) * scale
        if mask:
            attention_ = attention_.masked_fill(mask == 0, -1e9)
        attention_ = F.softmax(attention_, dim=-1)
        context = torch.matmul(attention_, Value)
        return context


class MultiHeadAttention(nn.Module):
    """ perform the attention function in parallel """

    def __init__(self, dim_model, num_head, dropout=0.0):
        """linearly project the queries, keys and values num_head times with different,
        learned linear projections to dk, dk and dv dimensions, respectively

        Args:
            dim_model(int): default 512
            num_head:
            dropout:
        Returns:
            None
        """
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0, "dim_model % num_head != 0"
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(in_features=dim_model, out_features=num_head * self.num_head)
        self.fc_K = nn.Linear(in_features=dim_model, out_features=num_head * self.num_head)
        self.fc_V = nn.Linear(in_features=dim_model, out_features=num_head * self.num_head)
        self.fc = nn.Linear(in_features=num_head * self.dim_head, out_features=dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(normalized_shape=dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        scale = K.size(-1) ** -0.5
        context = ScaledDotProductAttention.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_normal(out)
        return out


class PositionWiseFeedForward(nn.Module):
    """full connected feed-forward network"""
    def __init__(self, dim_model, hidden, dropout=0.0):
        """dimensionality of both input and output is d_model=512, inner layer has dimensionality 2048
        Args:
            dim_model(int):  dimensionality of model
            hidden(int):
            dropout(float):  probability of an element to be zeroed.

        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Embeddings(nn.Module):
    """convert the input tokens and output tokens to vector of dimension d_model"""
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
