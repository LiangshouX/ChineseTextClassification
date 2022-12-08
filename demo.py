import torch
from torch import nn
import torch.functional as F

class TextBILSTM(nn.Module):

    def __init__(self, vocab_size=5000, pinyin_size=5000):
        super(TextBILSTM, self).__init__()
        self.num_classes = 15
        self.learning_rate = config.learning_rate
        self.keep_dropout = config.keep_dropout
        self.char_embedding_size = config.char_embedding_size
        self.pinyin_embedding_size = config.pinyin_embedding_size
        self.l2_reg_lambda = config.l2_reg_lambda
        self.hidden_dims = config.hidden_dims
        self.char_size = char_size
        self.pinyin_size = pinyin_size
        self.rnn_layers = config.rnn_layers

        self.build_model()

    def build_model(self):
        # 初始化字向量
        self.char_embeddings = nn.Embedding(self.char_size, self.char_embedding_size)
        # 字向量参与更新
        self.char_embeddings.weight.requires_grad = True
        # 初始化拼音向量
        self.pinyin_embeddings = nn.Embedding(self.pinyin_size, self.pinyin_embedding_size)
        self.pinyin_embeddings.weight.requires_grad = True
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.char_embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.keep_dropout,
                                bidirectional=True)
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.num_classes)
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''

        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, char_id, pinyin_id):
        # char_id = torch.from_numpy(np.array(input[0])).long()
        # pinyin_id = torch.from_numpy(np.array(input[1])).long()

        sen_char_input = self.char_embeddings(char_id)
        sen_pinyin_input = self.pinyin_embeddings(pinyin_id)

        sen_input = torch.cat((sen_char_input, sen_pinyin_input), dim=1)
        # input : [len_seq, batch_size, embedding_dim]
        sen_input = sen_input.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(sen_input)
        # output : [batch_size, len_seq, n_hidden * 2]
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        return self.fc_out(atten_out)

