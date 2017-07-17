# -*- coding: utf-8 -*-
# @Author: Zhaoyin Zhang
# @Date:2017/7/11 10:22
# @Contact: 940942500@qq.com

import pandas as pd
from basicModel import LanguageModel
import data_utils
import tensorflow as tf
from keras.layers import LSTM, Dense, Embedding
import numpy as np
import sys
import time
#
# class LstmInput(object):
#     """The input data."""
#
#     def __init__(self, config, data, ):
#         self.batch_size = batch_size = config.batch_size
#         self.num_steps = num_steps = config.num_steps
#         self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
#         self.input_data, self.targets = data_utils.load_data(
#             config.FLAGS.trainset_dir)


class Config(object):

    batch_size = 64         # 每批训练数据的大小
    num_steps = 20          # 单条数据中序列的长度(sequence_length)
    max_epochs = 200        # 最大训练迭代次数
    early_stopping = 2      # 用于满足某个阈值，提前终止训练
    vocabulary_size = 5000  # 词典的规模大小
    embedding_size = 128    # 每个词向量的维度
    init_scale = 0.1        # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
    learning_rate = 0.01    # 学习速率(lr)
    max_grad_norm = 5       # 用于控制梯度膨胀
    num_layers = 2          # lstm神经网络的层数
    hidden_size = 200       # 隐藏层的大小
    keep_prob = 1.0         # 用于控制输入输出的dropout概率,防止过拟合







# build LSTM network
class LstmModel(LanguageModel):

    # def load_data(self, debug=False):
    #
    #     self.encoded_train =
    #     self.encoded_valid =
    #     self.encoded_test =



    def add_placeholders(self):

        self._input_data = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps], name='Input')
        self._targets = tf.placeholder(
            tf.int32, shape=[None, self.config.n_classes], name='Target')

        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def add_embedding(self):
        """Add embedding layer.

            Hint: This layer should use the input_placeholder to index into the
                  embedding.
            Hint: You might find tf.nn.embedding_lookup useful.
            Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
            Hint: Check the last slide from the TensorFlow lecture.
            Hint: Here are the dimensions of the variables you will need to create:

              L: (len(self.vocab), embed_size)

            Returns:
              inputs: List of length num_steps, each of whose elements should be
                      a tensor of shape (batch_size, embed_size).
            """
        # The embedding lookup is currently only implemented for the CPU

        with tf.device("/cpu:0"):
            # ===================addModel with tensroflow===================
            with tf.variable_scope("embedding_layer"):
                embedding_matrix = tf.get_variable("Embedding", [
                    self.config.vocabulary_size, self.config.embedding_size],trainable=True)
                inputs = tf.nn.embedding_lookup(
                    embedding_matrix, self._input_data)

                inputs = [
                    tf.squeeze(x,[1] ) for x in tf.split(1, self.config.num_steps, inputs)]
                return inputs

    def create_feed_dict(self, input_batch, label_batch, state, dropout):
        """
        :param input_batch:
        :param label_batch:
        :return:
        """

        feed_dict = {

            self.input_placeholder: input_batch,
            self.labels_placeholder: label_batch,
            self.initial_state: state,
            self.dropout_placeholder: dropout,
        }

        return feed_dict

    def add_model(self, input_data):
        """
        Args:
            input_data:
        Returns:
            logits:
        """

        # 输入层的dropout
        if self.dropout_placeholder < 1:
            with tf.variable_spope('InputDropout'):
                input_data = tf.nn.dropout(input_data,self.dropout_placeholder)

        # 建立lstm
        with tf.variable_scpoe("lstm") as scope:

            self.initial_state = tf.zeros([self.config.batch_size,self.config.hidden_size])

            states = self.initial_state
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size, forget_bias=0.0, state_is_tuple=True, )
            outputs, final_states = tf.nn.static_rnn(cell=lstm_cell, initial_state=states, inputs=input_data, sequence_length=self.config.num_steps, )


        # lstm输出层的dropout
        if self.dropout_placeholder < 1:
            with tf.variable_spope('OutputDropout'):
                outputs = tf.nn.dropout(outputs, self.dropout_placeholder)

        # 输出层的 softmax
        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable(
                name="softmax_w",
                shepe=[self.config.hidden_size, self.config.n_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01))
            softmax_b = tf.get_variable(
                name="softmax_b", shape=[self.config.n_classes],)

            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            return logits

    def add_loss_op(self, pred):

        with tf.name_scope("loss"):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self._targets))
            tf.add_to_collection('total_loss', cross_entropy)
            loss = tf.add_n(tf.get_collection('total_loss'))
            return loss

    def add_training_op(self, loss):

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op

    def run_epoch(self, session, input_data, input_labels, shuffle=True, verbose=True, train_op=None):
        """Runs an epoch of training.
            Trains the model for one-epoch.
            Args:
              sess: tf.Session() object
              input_data: np.ndarray of shape (n_samples, n_features)
              input_labels: np.ndarray of shape (n_samples, n_classes)
            Returns:
              average_loss: scalar. Average minibatch loss of model on epoch.
            """

        # And then after everything is built, start the training loop.

        dp = self.config.dropout
        state = self.initial_state.eval()
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(input_data) / self.config.batch_size

        tempData = data_utils.data_iterator(input_data, input_labels, batch_size=self.config.batch_size, label_size=self.config.n_classes, shuffle=shuffle)

        for step, (input_batch, label_batch) in enumerate(tempData):

            feed_dict = self.create_feed_dict(input_batch, label_batch, state, dropout = dp)

            loss, _ = session.run( [self.loss,  self.train_op], feed_dict=feed_dict)
            total_loss.append(loss)

        return np.mean(total_loss)

    def __init__(self, config):

        self.config = config
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.outputs = self.add_model(self.inputs)

        self.calculate_loss = self.add_loss_op(self.outputs)
        self.train_step = self.add_training_op(self.calculate_loss)



    def predict(self, sess, X, y=None):
        """Make predictions from the provided model."""
        losses = []
        results = []
        if np.any(y):
            data = data_utils.data_iterator(
                X,
                y,
                batch_size=self.config.batch_size,
                label_size=self.config.label_size,
                shuffle=False)
        else:
            data = data_utils.data_iterator(
                X,
                batch_size=self.config.batch_size,
                label_size=self.config.label_size,
                shuffle=False)
        for step, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, )
            if np.any(y):
                feed[self.labels_placeholder] = y
                loss, preds = sess.run(
                    [self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)
            else:
                preds = sess.run(self.predictions, feed_dict=feed)
            predicted_indices = preds.argmax(axis=1)
            results.extend(predicted_indices)
        return np.mean(losses), results



def testModel():
    config = Config()
    with tf.variable_scope('LSTM') as scope:
        model = LstmModel(config)
        scope.reuse_variables()

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0
        session.run(init)
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_pp = model.run_epoch(session, model.encoded_train, train_op=model.train_step)
            valid_pp = model.run_epoch(session, model.encoded_valid)

            print('Training perplexity: {}'.format(train_pp))
            print('Validation perplexity: {}'.format(valid_pp))

            if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, './ptb_rnnlm.weights')
            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))

            saver.restore(session, 'ptb_rnnlm.weights')
            test_pp = model.run_epoch(session, model.encoded_test)
            print('=-=' * 5)
            print('Test perplexity: {}'.format(test_pp))
            print('=-=' * 5)


