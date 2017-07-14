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

# build LSTM network
class LstmModel(LanguageModel):

    def add_placeholders(self):

        self._input_data = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps], name='Input')

        self._targets = tf.placeholder(
            tf.int32, shape=[None, self.config.n_classes], name='Target')

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
                    embedding_matrix, self.self._input_data)

                inputs = [
                    tf.squeeze(
                        x,
                        [1]) for x in tf.split(
                        1,
                        self.config.num_steps,
                        inputs)]
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

    def add_model(self, input_data,):
        """
        Args:
            input_data:
        Returns:
            logits:
        """

        # 输入层的dropout
        if self.config.keep_prob < 1:
            with tf.variable_spope('InputDropout'):
                input_data = tf.nn.dropout(input_data,self.config.keep_prob)

        # 建立lstm
        with tf.variable_scpoe("lstm") as scope:

            self.initial_state = tf.zeros([self.config.batch_size,self.config.hidden_size])

            states = self.initial_state
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                self.config.hidden_size, forget_bias=0.0, state_is_tuple=True,)
            outputs, states = tf.nn.static_rnn(
                initial_state=states,
                cell=lstm_cell, inputs=input_data,
                sequence_length=self.config.num_steps)

        # lstm输出层的dropout
        if self.config.keep_prob < 1:
            with tf.variable_spope('OutputDropout'):
                outputs = tf.nn.dropout(outputs, self.config.keep_prob)

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
            cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=pred, labels=self.labels_placeholder))
        # tf.add_to_collection('total_loss', cross_entropy)
        # loss = tf.add_n(tf.get_collection('total_loss'))
            return cross_entropy

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def run_epoch(
            self,
            sess,
            input_data,
            input_labels,
            shuffle=True,
            verbose=True):
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

        orig_X, orig_y = input_data, input_labels
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(orig_X) / self.config.batch_size

        # tf.estimator.inputs.pandas_input_fn()

        # dd = tf.estimator.inputs.numpy_input_fn(
        #     x={"x": np.array(orig_X)},
        #     y=np.array(orig_y),batch_size=self.config.batch_size,)

        dd = data_utils.data_iterator(
            orig_X,
            orig_y,
            batch_size=self.config.batch_size,
            label_size=self.config.n_classes,
            shuffle=shuffle)

        for step, (input_batch, label_batch) in enumerate(dd):

            feed_dict = self.create_feed_dict(input_batch, label_batch)

            loss, total_correct, _ = sess.run(
                [self.loss, self.correct_predictions, self.train_op],
                feed_dict=feed_dict)
            total_processed_examples += len(input_batch)
            total_correct_examples += total_correct
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()

        return np.mean(total_loss), total_correct_examples / \
            float(total_processed_examples)

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
