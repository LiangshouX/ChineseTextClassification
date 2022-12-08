import torch
import torch.nn as nn
import time
import numpy as np

from torch.utils.data import DataLoader
from Exp_DataSet import Corpus
from Exp_Model import BiLSTM_model, Transformer_model
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: Transformer_model, BiLSTM_model')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')

args = parser.parse_args()

def train():
    """
    完成一个 epoch 的训练
    """
    sum_true = 0
    sum_loss = 0.0

    max_valid_acc = 0

    # print(type(data_loader_train))  # <class 'torch.utils.data.dataloader.DataLoader'>
    model.train()
    for data in data_loader_train:
        # 选取对应批次数据的输入和标签
        # print(type(data))       # <'list'>
        batch_x, batch_y = data[0].to(device), data[1].to(device)
        # print(data[0].shape)    # data[0]: Tensor, [16,150];  data[1]:Tensor, [16]
        # print(data[1])
        # exit(0)

        # 模型预测
        y_hat = model(batch_x)

        loss = loss_function(y_hat, batch_y)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
        sum_true += torch.sum(y_hat == batch_y).float()
        sum_loss += loss.item()

    train_acc = sum_true / dataset.train.__len__()
    train_loss = sum_loss / (dataset.train.__len__() / batch_size)

    valid_acc = valid()

    if valid_acc > max_valid_acc:
        torch.save(model, "checkpoint.pt")

    print(
        f"epoch: [{epoch}/{epochs}], train loss: {train_loss:.4f}, train accuracy: {train_acc * 100:.2f}%, valid accuracy: {valid_acc * 100:.2f}%,\
            time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


def valid():
    """
    进行验证，返回模型在验证集上的 accuracy
    """
    sum_true = 0

    model.eval()
    with torch.no_grad():
        for data in data_loader_valid:
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            y_hat = model(batch_x)

            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            sum_true += torch.sum(y_hat == batch_y).float()

        return sum_true / dataset.valid.__len__()


def predict():
    """
    读取训练好的模型对测试集进行预测，并生成结果文件
    """
    results = []

    model = torch.load('checkpoint.pt').to(device)
    model.eval()
    with torch.no_grad():
        for data in data_loader_test:
            batch_x = data[0].to(device)

            y_hat = model(batch_x)
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat])

            results += y_hat.tolist()

    # 写入文件
    with open("predict.txt", "w") as f:
        for label_idx in results:
            label = dataset.dictionary.idx2label[label_idx][1]
            f.write(label + "\n")

if __name__ == '__main__':
    dataset_folder = 'E:\\ProgrammingFiles\\Python\\NLP\\shortTextClassification\\tnews_public'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # -----------------------------------------------------begin-----------------------------------------------------#
    # 以下为超参数，可根据需要修改
    embedding_dim = 100
    max_sent_len = 150
    batch_size = 16
    epochs = 20
    lr = 1e-4
    # ------------------------------------------------------end------------------------------------------------------#

    dataset = Corpus(dataset_folder, max_sent_len)

    vocab_size = len(dataset.dictionary.tkn2word)
    print("vocab_size: ", vocab_size)   # 5082

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    # -----------------------------------------------------begin-----------------------------------------------------#
    # 可修改选择的模型以及传入的参数
    if args.model == 'Transformer_model':
        if args.embedding == 'random':
            model = Transformer_model(vocab_size).to(device)  # 设置模型
        else:
            # embedding_weight = load_vector()
            model = Transformer_model(vocab_size, embedding_weight=dataset.weight_matrix).to(device)
        # print(model)
    elif args.model == 'BiLSTM_model':
        if args.embedding == 'random':
            model = BiLSTM_model(vocab_size).to(device)
        else:
            model = BiLSTM_model(vocab_size, embedding_weight=dataset.weight_matrix).to(device)
    else:
        model = None
        exit(9999)
        # print(model)
    # ------------------------------------------------------end------------------------------------------------------#
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # 设置优化器

    # 进行训练
    for epoch in range(epochs):
        train()

    # 对测试集进行预测
    predict()
