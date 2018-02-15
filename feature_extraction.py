import glob
import librosa
import numpy as np
import warnings
import progress_printer as pp
import time
import platform

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def extract_filename(file_and_path):
    if platform.system() == 'Windows':
        return file_and_path.split('/')[2].split('\\')
    return file_and_path.split('/')[3]


def parse_audio_files(filenames, progress_printer):
    rows = len(filenames)
    features, labels, groups = np.zeros((rows, 193)), np.zeros((rows, 10)), np.zeros((rows, 1))
    i = 0
    for fn in filenames:
        start = time.time()
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            y_col = int(extract_filename(fn).split('-')[1])
            group = int(extract_filename(fn).split('-')[0])
        except Exception as e:
            print("Error loading " + str(fn) + ". Reason: " + str(e))
        else:
            features[i] = ext_features
            labels[i, y_col] = 1
            groups[i] = group
            i += 1
        progress_printer.register_progress_time(time.time() - start)
    return features, labels, groups


audio_files = []
for i in range(1, 11):
    audio_files.extend(glob.glob('UrbanSound8K/audio/fold%d/*.wav' % i))

number_of_files = len(audio_files)
progress_printer = pp.ProgressPrinter(number_of_files)
progress_printer.start()
running = True
print("Started processing " + str(number_of_files) + " files\n")
for i in range(9):
    files = audio_files[i * 1000: (i + 1) * 1000]
    if len(files):
        X, y, groups = parse_audio_files(files, progress_printer)
        for r in y:
            if np.sum(r) > 1.5:
                print('error occured')
                break
        print("\nSaving data to file " + ('./processed_audio/urban_sound_%d\t\t\t\t' % i))
        np.savez('./processed_audio/urban_sound_%d' % i, X=X, y=y, groups=groups)
    else:
        progress_printer.kill()
        progress_printer.join()
        print("Processing completed")
        running = False
        break

if not running:
    progress_printer.kill()
    progress_printer.join()
    print("Processing completed")
