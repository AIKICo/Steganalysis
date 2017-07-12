import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification
#from dbn import SupervisedDBNClassification


def loadData(filename,instanceCol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = [];
    y = [];
    for row in file_reader:
        x.append(row[0:instanceCol]);
        y.append(row[-1]);
    return np.array(x[1:]).astype((np.float32)), np.array(y[1:]).astype(np.int);



X,Y = loadData('/home/mohammad/Documents/python/Steganalysis/feature.csv',99);
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0);

classifier = SupervisedDBNClassification(hidden_layers_structure=[1024,1024],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=10,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))