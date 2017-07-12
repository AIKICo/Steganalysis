import os, wave, struct, sys, math;
import numpy as np;
import scipy.io.wavfile as wav
import soundfile as sf
from sklearn.cluster import KMeans
from pandas import Series, DataFrame


def hfd(X, Kmax):
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


def feature_extraction(path, label):
    root, dirs, files = next(os.walk(path));
    sr = [];
    x = [];
    xf = [];
    for file in files:
        if file.lower().endswith('.wav'):
            sr_value, x_value = wav.read(root + '/' + file,'r');
            sr.append(sr_value);
            x.append(x_value);
            f = [];
            length = len(x_value);
            window_hop_length = 0.02  # 20ms
            overlap = int(sr_value * window_hop_length);
            window_size = 0.05  # 5 ms
            framesize = int(window_size * sr_value);
            number_of_frames = int(length / overlap);
            frames = np.ndarray((number_of_frames, framesize));

            # Signal Framing
            for k in range(0, number_of_frames):
                for i in range(0, framesize):
                    if ((k * overlap + i) < length):
                        frames[k][i] = x_value[k * overlap + i]
                    else:
                        frames[k][i] = 0

            # Transfer To Fractal Dimension
            for k in range(0, number_of_frames):
                f.append(hfd(frames[k], 6));
            xf.append(f);

    Features = DataFrame();
    for vector in xf:
        kmeans = KMeans(n_clusters=100, random_state=0).fit(DataFrame(vector));
        Features = Features.append(DataFrame(kmeans.cluster_centers_).transpose());

    # Add Label Column
    Features['label'] = label;

    # Export Dataframe To CSV
    Features.to_csv('/home/mohammad/Documents/python/Steganalysis/feature.csv', mode='a', header=False, index=False);


feature_extraction('/home/mohammad/Documents/python/Steganalysis/clean', 0);
feature_extraction('/home/mohammad/Documents/python/Steganalysis/steg', 1);
