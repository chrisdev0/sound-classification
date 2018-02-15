import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import progress_printer as pp

# sound_names = ["air conditioner", "car horn", "children playing", "dog bark", "drilling", "engine idling","gun shot", "jackhammer", "siren", "street music"]

sound_data = np.load('processed_audio/urban_sound.npz')
X_data = sound_data['X']
y_data = sound_data['y']

X_sub, X_test, y_sub, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2)

print(len(X_train), len(X_val), len(X_test), len(y_train), len(y_val), len(y_test))
print(X_train.shape, y_train.shape)

# training_epochs = 6000
training_epochs = 6000
n_dim = 193
n_classes = 10
n_hidden_units_one = 300
n_hidden_units_two = 200
n_hidden_units_three = 100
learning_rate = 0.01
sd = 1 / np.sqrt(n_dim)

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name="w1")
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="w2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="w3")
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)

W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="w")
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
y_ = tf.nn.softmax(tf.matmul(h_3, W) + b)

init = tf.initialize_all_variables()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

progress_output = pp.ProgressPrinter(training_epochs)
cost_history = np.empty(shape=[1], dtype=float)
with tf.Session() as sess:
    fw = tf.summary.FileWriter('./summary/summary', sess.graph)
    sess.run(init)
    progress_output.start()
    for epoch in range(training_epochs):
        start = time.time()
        _, cost = sess.run([optimizer, cost_function], feed_dict={X: X_sub, Y: y_sub})
        progress_output.register_progress_time(time.time() - start)
        cost_history = np.append(cost_history, cost)
    progress_output.kill()
    progress_output.join()
    print('\nValidation accuracy: ', round(sess.run(accuracy, feed_dict={X: X_test, Y: y_test}), 3))
    saver.save(sess, "./models/model_321.ckpt")

plt.plot(cost_history)
plt.show()
