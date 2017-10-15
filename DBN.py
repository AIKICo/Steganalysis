import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
from dbn.tensorflow import SupervisedDBNClassification
# from dbn import SupervisedDBNClassification


def loaddata(filename,instanceCol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = []
    y = []
    for row in file_reader:
        x.append(row[0:instanceCol])
        y.append(row[-1])
    return np.array(x[1:]).astype((np.float32)), np.array(y[1:]).astype(np.int)


def fractal_modeldata(filename):
    scores = []
    print(filename)
    X, Y = loaddata(filename, 99)

    for i in range(1):
        np.random.seed(13)
        indices = np.random.permutation(1000)
        test_size = int(0.1 * len(indices))
        X_train = X[indices[:-test_size]]
        Y_train = Y[indices[:-test_size]]
        X_test = X[indices[-test_size:]]
        Y_test = Y[indices[-test_size:]]
        # relu, sigmoid
        classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                                 learning_rate_rbm=0.05,
                                                 learning_rate=0.2,
                                                 n_epochs_rbm=30,
                                                 n_iter_backprop=4000,
                                                 batch_size=16,
                                                 activation_function='sigmoid',
                                                 dropout_p=0.1,
                                                 verbose=0)
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        scores.append(accuracy_score(Y_test, Y_pred))
        print(classification_report(Y_test, Y_pred))

    print('All Accuracy Scores in Cross: ' + str(scores))
    print('Mean Accuracy Scores: ' + str(np.mean(scores)))


if __name__ == '__main__':
    fractal_modeldata('D:\\Databases\\PDA\\CSV\\feature(Fractal-70-30-1400b).csv')
