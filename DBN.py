import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
from dbn.tensorflow import SupervisedDBNClassification
#from dbn import SupervisedDBNClassification
from sklearn.model_selection import cross_val_score


def loaddata(filename,instanceCol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = []
    y = []
    for row in file_reader:
        x.append(row[0:instanceCol])
        y.append(row[-1])
    return np.array(x[1:]).astype((np.float32)), np.array(y[1:]).astype(np.int)

scores = []
X,Y = loaddata('/home/mohammad/Documents/python/Steganalysis/feature.csv', 99)

for i in range(3):
    print('Cross ' + str(i))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # relu, sigmoid
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                             learning_rate_rbm=0.06,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=1000,
                                             batch_size=32,
                                             activation_function='sigmoid',
                                             dropout_p=0.2,
                                             verbose=0);
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    scores.append(accuracy_score(Y_test, Y_pred))
    print('Accuracy: %f' % accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))

print('All Accuracy Scores in Cross: ' + str(scores))
print('Mean Accuracy Scores: ' + str(np.mean(scores)))