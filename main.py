# -*- coding: utf-8 -*-
# @Author: Zhaoyin Zhang
# @Date:2017/7/4 14:42
# @Contact: 940942500@qq.com
import pandas as pd
from data_utils import cut, getLabels
from sklearn.model_selection import train_test_split
from modelTrain import Models
from modelLstm import LstmModel
import pandas as pd
import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "indicates the checkpoint dircotry")
flags.DEFINE_string("trainset_dir", "D:/company_DATA_03.csv", "indicates the trainset dircotry")

FLAGS = flags.FLAGS


"""
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
"""


class Config(object):

    batch_size = 64        # 每批训练数据的大小
    num_steps = 20         # 单条数据中序列的长度(sequence_length)
    max_epochs = 200       # 最大训练迭代次数
    early_stopping = 2     # 用于满足某个阈值，提前终止训练
    vocabulary_size = 5000 # 词典的规模大小
    embedding_size = 128   # 每个词向量的维度
    init_scale = 0.1       # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
    learning_rate = 0.01   # 学习速率(lr)
    max_grad_norm = 5      # 用于控制梯度膨胀
    num_layers = 2         # lstm神经网络的层数
    hidden_size = 200      # 隐藏层的大小
    keep_prob = 1.0        # 用于控制输入输出的dropout概率,防止过拟合



def main(_):

    config = Config()
    model = Models(config)
    tf = model.trainTf()
    clf = model.trianModel()
    testData = pd.read_csv(config.FLAGS.trainfile_dir)
    testData["cut"] = testData["scapeOfBesiness"].apply(cut)
    X_test = tf.transform(testData["cut"])
    y_test = clf.predict(X_test)
    testData["secInduCode"] = pd.DataFrame(y_test)
    labels = pd.DataFrame.from_dict(getLabels())
    testData = testData.merge(labels, on="secInduCode", how="left")

    testData.to_csv(
        "D:/zzy@creditease.cn/Company22.csv", index=None, encoding="utf8")

    print(testData[["companyName", "secnduName"]].head())

    # print("训练集和测试集切分...")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=43)
    # X_train_L = tfidf.transform(X_train)
    # X_test_L = tfidf.transform(X_test)


if __name__ == '__main__':
    tf.app.run()
