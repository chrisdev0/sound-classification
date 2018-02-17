import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import progress_printer as pp
import tensor_config as tc

# sound_names = ["air conditioner", "car horn", "children playing", "dog bark", "drilling", "engine idling","gun shot", "jackhammer", "siren", "street music"]

sound_data = np.load('processed_audio/urban_sound.npz')
X_data = sound_data['X']
y_data = sound_data['y']

X_sub, X_test, y_sub, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2)

n_dim = 193
n_classes = 10

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

config = tc.TensorConfig(learning_rate=0.0001, hidden_units=[300, 200, 100],
                         config_number=3, training_epochs=1000000,
                         X=X, Y=Y, n_dim=n_dim, n_classes=n_classes)

# config = tc.TensorConfig(learning_rate=0.01, hidden_units=[300, 200, 100],
#                         config_number=3, training_epochs=6000,
#                         X=X, Y=Y, n_dim=n_dim, n_classes=n_classes)


saver = tf.train.Saver()

progress_output = pp.ProgressPrinter(config.training_epochs)
cost_history = np.empty(shape=[1], dtype=float)
with tf.Session() as sess:
    fw = tf.summary.FileWriter('./summary/summary', sess.graph)
    sess.run(tf.global_variables_initializer())
    progress_output.start()
    for epoch in range(config.training_epochs):
        start = time.time()
        _, cost = sess.run([config.optimizer, config.cost_function], feed_dict={X: X_sub, Y: y_sub})
        progress_output.register_progress_time(time.time() - start)
        cost_history = np.append(cost_history, cost)
    progress_output.kill()
    progress_output.join()
    print('\nValidation accuracy: ', round(sess.run(config.accuracy, feed_dict={X: X_test, Y: y_test}), 3))
    saver.save(sess, "./models/model_321.ckpt")

plt.plot(cost_history)
plt.show()