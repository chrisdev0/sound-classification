import tensorflow as tf
import numpy as np


def get_wbh_1(n_dim, hidden_units, sd, X):
    W_1 = tf.Variable(tf.random_normal([n_dim, hidden_units[0]], mean=0, stddev=sd), name="w1")
    b_1 = tf.Variable(tf.random_normal([hidden_units[0]], mean=0, stddev=sd), name="b1")
    h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)
    return W_1, b_1, h_1


def get_wbh_2(hidden_units, h_1, sd):
    W_2 = tf.Variable(tf.random_normal([hidden_units[0], hidden_units[1]], mean=0, stddev=sd), name="w2")
    b_2 = tf.Variable(tf.random_normal([hidden_units[1]], mean=0, stddev=sd), name="b2")
    h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)
    return W_2, b_2, h_2


def get_wbh_3(hidden_units, h_2, sd):
    W_3 = tf.Variable(tf.random_normal([hidden_units[1], hidden_units[2]], mean=0, stddev=sd), name="w3")
    b_3 = tf.Variable(tf.random_normal([hidden_units[2]], mean=0, stddev=sd), name="b3")
    h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)
    return W_3, b_3, h_3


class TensorConfig:
    def __init__(self, learning_rate, hidden_units, config_number, training_epochs, X, Y, n_dim, n_classes):
        if config_number < 1 or config_number > 3:
            raise Exception
        sd = 1 / np.sqrt(n_dim)
        self.training_epochs = training_epochs
        self.n_dim = n_dim
        self.n_classes = n_classes
        W_1, b_1, h_1 = get_wbh_1(n_dim, hidden_units, sd, X)
        W_2, b_2, h_2 = get_wbh_2(hidden_units, h_1, sd)
        W_3, b_3, h_3 = get_wbh_3(hidden_units, h_2, sd)

        b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
        W = tf.Variable(tf.random_normal([hidden_units[config_number - 1], n_classes], mean=0, stddev=sd), name="w")

        if config_number == 1:
            y_ = tf.nn.softmax(tf.matmul(h_1, W) + b)
        elif config_number == 2:
            y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)
        elif config_number == 3:
            y_ = tf.nn.softmax(tf.matmul(h_3, W) + b)

        self.cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost_function)

        self.correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class TensorTestConfig:
    def __init__(self, config_number, hidden_units, n_dim, n_classes, X):
        if config_number < 1 or config_number > 3:
            raise Exception("incorrect config number")
        sd = 1 / np.sqrt(n_dim)

        W_1, b_1, h_1 = get_wbh_1(n_dim, hidden_units, sd, X)
        W_2, b_2, h_2 = get_wbh_2(hidden_units, h_1, sd)
        W_3, b_3, h_3 = get_wbh_3(hidden_units, h_2, sd)

        W = tf.Variable(tf.random_normal([hidden_units[config_number - 1], n_classes], mean=0, stddev=sd), name="w")
        b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
        if config_number == 1:
            z = tf.matmul(h_1, W) + b
        elif config_number == 2:
            z = tf.matmul(h_2, W) + b
        elif config_number == 3:
            z = tf.matmul(h_3, W) + b
        else:
            raise Exception("incorrect config number")

        self.y_sigmoid = tf.nn.sigmoid(z)
        self.y_ = tf.nn.softmax(z)
