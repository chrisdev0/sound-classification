import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
               "gun shot","jackhammer","siren","street music"]
#sound_data = np.load('urban_sound_train.npz')
sound_data = np.load('urban_sound.npz')
X_data = sound_data['X']
y_data = sound_data['y']
groups = sound_data['groups']





from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in gss.split(X_data, y_data, groups=groups):
    X_train = X_data[train_idx]
    y_train = y_data[train_idx]
    groups_train = groups[train_idx]

    X_val = X_data[val_idx]
    y_val = y_data[val_idx]
    groups_test = groups[val_idx]

len(X_train), len(X_val), len(y_train), len(y_val)    


X_train.shape, y_train.shape


np.intersect1d(groups_train, groups_test)


training_epochs = 1000
n_dim = 193
n_classes = 10
learning_rate = 0.001


g1 = tf.Graph()
with g1.as_default():
    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    c1 = tf.layers.conv2d(tf.reshape(X, [-1, 1, n_dim, 1]), 50, (1, 5), padding='same', 
                          activation=tf.nn.sigmoid, name="c1")
    p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[1, 2], strides=2)
    c2 = tf.layers.conv2d(tf.reshape(p1, [-1, 1, 96, 50]), 100, (1, 5), padding='same', 
                          activation=tf.nn.sigmoid, name="c2")
    p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[1, 2], strides=2)

    h_p = tf.reshape(p2, [-1, 48*100])

    h_1 = tf.layers.dense(inputs=h_p, units=1000, activation=tf.nn.sigmoid,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")

    y_hat = tf.layers.dense(inputs=h_1, units=n_classes,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                            name="h4")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_hat))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


config = tf.ConfigProto(
	device_count = {'GPU': 0}
)

cost_history = []
with tf.Session(graph=g1, config=config) as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _, cost = sess.run([optimizer, loss], feed_dict={X: X_train, Y: y_train})
        cost_history.append(cost)
    
    print('Validation accuracy: ', round(sess.run(accuracy, feed_dict={X: X_val, Y: y_val}) , 3))
    print('Last cost: ', round(cost_history[-1], 3))
#plt.plot(cost_history)
#plt.show()



cost_history = []
sess = tf.Session(graph=g1,config=config)
tf.reset_default_graph()
sess.run(init)
for epoch in range(training_epochs):
    _, cost = sess.run([optimizer, loss], feed_dict={X: X_data, Y: y_data})
    print(str(epoch) + "/" + str(training_epochs) + "\tCost: " + str(cost))
    cost_history.append(cost)
    
print('Last cost: ', round(cost_history[-1], 3))
plt.plot(cost_history)
plt.show()
saver.save(sess, "model_adam.ckpt")
sess.close()

