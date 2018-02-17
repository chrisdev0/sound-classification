"""
 Start the web server by running the following
 Linux/MacOS
    export FLASK_APP=urban_sound_classifier.py
    flask run
 Windows
    set FLASK_APP=urban_sound_classifier.py
    flask run

 ... Running on http://127.0.0.1:5000
"""

import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, request
import os
import tensor_config as tc


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


n_dim = 193
n_classes = 10
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])
tensor_config = tc.TensorTestConfig(3, [300, 200, 100], n_dim, n_classes, X)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, './models/model_321.ckpt')

app = Flask(__name__, static_folder='./')
app.config['UPLOAD_FOLDER'] = './upload'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return ''
    file = request.files['file']
    if file.filename == '':
        return ''
    audio_file = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(audio_file)
    mfccs, chroma, mel, contrast, tonnetz = extract_feature(audio_file)
    x_data = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    y_hat, sigmoid = sess.run([tensor_config.y_, tensor_config.y_sigmoid], feed_dict={X: x_data.reshape(1, -1)})
    index = np.argmax(y_hat)
    print(sigmoid)
    return '%d' % (index)


@app.route('/send_sound', methods=['GET'])
def upload_file_page():
    return app.send_static_file("upload.html")
