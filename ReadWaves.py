import os, wave
import numpy as np
import scipy.io.wavfile as wav
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from pandas import Series, DataFrame,read_csv


def hfd(X, Kmax):
    try:
        L = []
        x = []
        N = len(X)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(float(1) / k), 1])
            (p, r1, r2, s) = np.linalg.lstsq(x, L)
        return p[0]
    except Exception as e:
        print('Excepton: ' + str(e));
        return 0;


def feature_extraction(infile,path, label):
    root, dirs, files = next(os.walk(path))
    sr = []
    x = []
    xf = []
    file_index=1
    for file in files:
        if file.lower().endswith('.wav'):
            sr_value, x_value = wav.read(root + '/' + file, 'r')
            sr.append(sr_value)
            x.append(x_value)
            f = []
            length = len(x_value)
            window_hop_length = 0.02  # 2ms = 0.02
            overlap = int(sr_value * window_hop_length)
            window_size = 0.05  # 5 ms = 0.05
            framesize = int(window_size * sr_value)
            number_of_frames = int(length / overlap)
            frames = np.ndarray((number_of_frames, framesize))

            # Signal Framing and Transfer To Fractal Dimension
            for k in range(0, number_of_frames):
                for i in range(0, framesize):
                    if (k * overlap + i) < length:
                        frames[k][i] = x_value[k * overlap + i]
                    else:
                        frames[k][i] = 0
                f.append(hfd(frames[k], 6))

            xf.append(f)
            print('FileName: ' + file + ' Row: ' + str(file_index) + ' Of ' + str(len(files)))
            file_index = file_index + 1

    features = DataFrame()
    vector_index = 1
    for vector in xf:
        try:
            km = KMeans(n_clusters=100, random_state=42).fit(DataFrame(vector))
            features = features.append(DataFrame(km.cluster_centers_).transpose())
            print('Vector Cluster is Success: ' + str(vector_index) + ' Vector Length: ' + str(len(vector)))
            vector_index = vector_index + 1
        except Exception as e:
            print('Exception in Clustering:' + str(e))

    # Add Label Column
    features['label'] = label

    # Export Data frame To CSV
    features.to_csv(infile, mode='a', header=False, index=False)


if __name__ == '__main__':
    csv_filename = 'D:\\Databases\\PDA\\CSV\\feature(70-30-1400b).csv'
    feature_extraction(csv_filename, 'D:\\Databases\\PDA\\Normal', 0)
    feature_extraction(csv_filename, 'D:\\Databases\\PDA\\StegHide', 1)
