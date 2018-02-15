import glob
import numpy as np

X = np.empty((0, 193))
y = np.empty((0, 10))
groups = np.empty((0, 1))
npz_files = glob.glob('./processed_audio/urban_sound_?.npz')
print("Merging " + str(len(npz_files)) + " files...")
for fn in npz_files:
    print("Loading " + fn)
    data = np.load(fn)
    X = np.append(X, data['X'], axis=0)
    y = np.append(y, data['y'], axis=0)
    groups = np.append(groups, data['groups'], axis=0)


for r in y:
    if np.sum(r) > 1.5:
        print(r)

print("Merging complete. Saving to file " + './processed_audio/urban_sound')
np.savez('./processed_audio/urban_sound', X=X, y=y, groups=groups)
print("Done")