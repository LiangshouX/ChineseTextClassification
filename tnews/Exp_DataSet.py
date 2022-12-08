import os
import json
import numpy as np
import torch
import jieba
from torch.utils.data import TensorDataset


class Dictionary(object):
    def __init__(self, path):
        self.word2tkn = {}
        self.tkn2word = []

        self.label2idx = {}
        self.idx2label = []

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    """
    完成对数据集的读取和预处理，处理后得到所有文本数据的 tokens 表示及相应的标签。

    该类适用于任务一、任务二，若要完成任务三，需对整个类进行简化，只需调用预训练 tokenizer 即可将文本的数据全部转为 tokens 的数据。
    """

    def __init__(self, path, max_sent_len, pre_train=False):
        """读取数据集
        Args:
            path(str):  数据集的根目录
            max_sent_len(int):  超参数，读取句子的最大长度
            pre_train(bool): 是否采用预训练
        Returns:
            A Corpus object
        """
        self.dictionary = Dictionary(path)

        self.max_sent_len = max_sent_len
        self.pre_train = pre_train
        self.weight_matrix = None

        self.train = self.tokenize(os.path.join(path, 'train.json'))  # 训练集， TensorDataset
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))  # 验证集， TensorDataset
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)  # 测试集， TensorDataset

        # -----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embeddings 的映射矩阵

        # ------------------------------------------------------end------------------------------------------------------#

    def tokenize(self, path, test_mode=False):
        """
        将数据集的每一个 sent 都转化成对应的 tokens.
        Args:
            path(str):  数据集路径以及文件名
            test_mode(bool):
        Returns:
            Test_mode: 数据Tensor 的DataTensor联合

            not Test_mode: 数据Tensor 和标签Tensor 的DataTensor联合
        """
        with open(path, 'r', encoding='utf8') as f:
            word_dic = {}
            one_word_index = 0

            if test_mode:
                idss = []
                for line in f:
                    one_data = json.loads(line)  # 读取一条数据

                    sent = one_data['sentence']
                    # -----------------------------------------------------begin-----------------------------------------------------#
                    # 若采用预训练的 embedding, 可以在此处对 sent 分词操作

                    # ------------------------------------------------------end------------------------------------------------------#
                    # 向词典中添加词
                    for word in sent:
                        self.dictionary.add_word(word)

                    ids = []
                    for word in sent:
                        ids.append(self.dictionary.word2tkn[word])
                    idss.append(self.pad(ids))

                idss = torch.tensor(np.array(idss))

                return TensorDataset(idss)

            else:
                idss = []
                labels = []
                for line in f:
                    one_data = json.loads(line)  # 读取一条数据

                    sent = one_data['sentence']
                    label = one_data['label']
                    # print(sent)
                    # -----------------------------------------------------begin-----------------------------------------------------#
                    # 若要采用预训练的 embedding, 需在此处对 sent 进行分词
                    if self.pre_train:
                        seg_sent = jieba.cut(sent)
                        for words in seg_sent:
                            if words in word_dic:
                                pass
                            else:
                                word_dic[word] = one_word_index
                                one_word_index += 1
                    # ------------------------------------------------------end------------------------------------------------------#
                    # 向词典中添加词
                    for word in sent:
                        self.dictionary.add_word(word)

                    ids = []
                    for word in sent:
                        ids.append(self.dictionary.word2tkn[word])
                    idss.append(self.pad(ids))
                    labels.append(self.dictionary.label2idx[label])

                if self.pre_train:
                    self.weight_matrix = load_vector(word_index=word_dic)

                idss = torch.tensor(np.array(idss))
                labels = torch.tensor(np.array(labels)).long()

                return TensorDataset(idss, labels)

    def pad(self, origin_sent):
        """
        padding: 将一个 sentence 补 0 至预设的最大句长 self.max_sent_len
        Args:
            origin_sent(list): 读取的原始的句子
        Returns:
            补齐后的sentence
        """
        if len(origin_sent) > self.max_sent_len:
            return origin_sent[:self.max_sent_len]
        else:
            return origin_sent + [0 for _ in range(self.max_sent_len - len(origin_sent))]


def load_vector(word_index):
    """先借助分词器生成一个word_index，由这个word_index在预训练的词向量中提取出需要的embedding_matrix
        Args:
            word_index: 分词器生成的word_index, 第一个元素为word, 第二个元素为索引
        Returns:
            embedding_matrix: 按照分词器给出的索引组成的词嵌入矩阵
    """
    EMBEDDING_FILE = "sgns.baidubaike.bigram-char"
    max_features = 150
    with open(EMBEDDING_FILE) as f:
        # 用于将embedding的每行的第一个元素word和后面为float类型的词向量分离出来。
        def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

        # 将所有的word作为key，numpy数组作为value放入字典
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    # 取出所有词向量
    all_embs = np.stack(embeddings_index.values())
    # 计算所有元素的均值和标准差
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    # 每个词向量的长度
    embed_size = all_embs.shape[1]

    # len(word_index)是数据集中的不同词的数量， word_index是在加载数据集时，在数据集上用分词器Tokenizer得到的。
    # max_feature是最大特征数量， 手动设置
    nb_words = min(max_features, len(word_index))

    # 在高斯分布上采样， 利用计算出来的均值和标准差， 生成和预训练好的词向量相同分布的随机数组成的ndarray （假设预训练的词向量符合高斯分布）
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    # 给embedding_matrix中的一些行用预训练的词向量进行赋值。利用预训练的词向量的均值和标准差为数据集中的词随机初始化词向量。
    # 然后再使用预训练词向量中的词去替换随机初始化数据集的词向量。
    for word, i in word_index.items():
        # 如果已经修改完了所有用到的词的词向量,就跳过本次循环
        if i >= max_features: continue
        # 如果dict的key中包括word， 就返回其value。 否则返回none。
        embedding_vector = embeddings_index.get(word)
        # 如果返回不为none，说明这个词在数据集中和训练词向量的数据集中都出现了，可以使用预训练的词向量替换随机初始化的词向量
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    # 经过上面的循环，返回的 embedding_matrix 中每行都是根据分词器的索引进行赋值的，因此之后可以直接根据词的索引取对应的词向量
    return embedding_matrix


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset_folder = 'E:\\ProgrammingFiles\\Python\\NLP\\shortTextClassification\\tnews_public'
    dataset = Corpus(dataset_folder, 150)
    # print(dataset.train)
    data_loader_train = DataLoader(dataset=dataset.train, batch_size=16, shuffle=True)
    for data in data_loader_train:
        print(data[0].shape)
        print(data[0][0])
        exit(0)
