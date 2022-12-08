import copy

import torch.nn as nn
import torch.nn.functional as F
import torch as torch
import math

# from Transform_Modules import *

class PositionalEncoding(nn.Module):
    """《Attention is all you need》提出的方法，用以记录词与词之间的顺序关系
    Add the position encoding to the input embeddings at the bottom of the encoder he decoder stacks
    The positional encoding have the same dimension d_model as the embeddings
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # PE(pos,2i) = sin(pos/10000^{2i/d_model})  ----pos:position, i: dimension
        pe[:, 0::2] = torch.sin(position * div_term)
        # PE(pos,2i+1) = con(pos/10000^{2i/d_model})
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        """convert the input tokens and output tokens to vector of dimension d_model
        Args:
            d_model(int):
            vocab(int): num_embeddings
        Returns:
            A Tensor
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, num_inp=512, num_token=150, num_hid=2048, num_head=8, num_layers=2, dropout=0.5,
                 embedding_weight=None):
        """Transformer model"""
        super(Transformer_model, self).__init__()
        self.num_classes = 15  # 类别数，Tnews的类别数是15
        # -----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计词嵌入层
        if embedding_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
            self.embed.weight.requires_grad = False
        else:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_inp)

        # ------------------------------------------------------end------------------------------------------------------#
        self.pos_encoder = PositionalEncoding(d_model=num_inp, max_len=num_token)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=num_inp, nhead=num_head, dim_feedforward=num_hid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=num_layers)
        # -----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 transformer 隐藏层数据的处理和选择方法

        # 请自行设计分类器
        self.dim_classification = 512
        self.classifier = nn.Linear(num_inp * num_token, self.num_classes)

        # ------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)
        # print(x.shape)  # torch.Size([16, 150, 512])
        # x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # print(x.shape)  # [150, 16, 512]
        # x = x.permute(1, 0, 2)
        # print(x.shape)  # torch.Size([16, 150, 512]), [batch_size, pad_size, ]
        # -----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类

        x = x.reshape(x.shape[0], -1)
        # x = self.fc1(x)
        x = self.classifier(x)
        # x = F.softmax(x, dim=-1)

        # ------------------------------------------------------end------------------------------------------------------#
        return x


class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, num_inp=100, num_token=150, num_hid=80, num_layers=1, dropout=0.2,
                 embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        # -----------------------------------------------------begin-----------------------------------------------------#
        # 自行设计词嵌入层
        # self.embed = Embeddings(num_inp, vocab_size)
        if embedding_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
            self.embed.weight.requires_grad = False
        else:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_inp, padding_idx=vocab_size-1)

        # ------------------------------------------------------end------------------------------------------------------#
        self.lstm = nn.LSTM(input_size=num_inp, hidden_size=num_hid, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attention_layer = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(inplace=True)
        )
        # -----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bi_lstm 隐藏层数据的处理和选择方法
        self.num_classes = 15
        self.classifier = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_hid, self.num_classes)
        )
        # 请自行设计分类器

        # ------------------------------------------------------end------------------------------------------------------#

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        """attention
        Args:
            lstm_out(Tensor):[batch_size, len_seq, n_hidden * 2]
            lstm_hidden(Tensor): [batch_size, num_layers * num_directions, n_hidden]
        Returns:
            result(Tensor):[batch_size, n_hidden]
        """
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)

        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]

        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)

        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)

        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # print(atten_w.shape)    # [16, 1, 80]
        # exit(0)

        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)

        # atten_context [batch_size, 1, time_step]
        # print(m.shape)                           # ([150, 16, 80]
        # exit(0)

        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        x = self.embed(x)
        # print(x.size())     # [16, 150, 100], [batch_size, time_step, hidden_dims]
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        # print(x.size())     # [16, 150, 160], [batch_size, seq_len, num_hid * 2]
        # x = x.permute(1, 0, 2)
        # x = self.dropout(x)
        # -----------------------------------------------------begin-----------------------------------------------------#
        # 对 bi_lstm 的隐藏层输出进行处理和选择，并完成分类

        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # print(final_hidden_state.shape)     # [16, 2 80]
        # print(x.shape)
        # exit(0)
        atten_out = self.attention_net_with_w(x, final_hidden_state)
        out = self.classifier(atten_out)
        # ------------------------------------------------------end------------------------------------------------------#

        return out
