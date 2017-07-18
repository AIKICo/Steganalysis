import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from dbn.tensorflow import SupervisedDBNClassification
#  from dbn import SupervisedDBNClassification


def loaddata(filename,instanceCol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = []
    y = []
    z = []
    for row in file_reader:
        x.append(row[0:instanceCol])
        y.append(row[-1])
        z.append(row)
    return np.array(x[1:]).astype(np.float64), np.array(y[1:]).astype(np.int),np.array(z[:]).astype(np.float64)

scores = []
AttributeCount = 99;
X,Y,Z = loaddata('/home/mohammad/Documents/python/Steganalysis/feature(4000-4000-90b).csv', AttributeCount)

Data_NotSteg = Z[Z[:,AttributeCount + 1] == 0]
Data_Steg = Z[Z[:,AttributeCount + 1] == 1][0:1200,:]

print(Data_NotSteg.shape)
print(Data_Steg.shape)
for i in range(3):
    print('Iteration ' + str(i+1))
    X_train = np.append(Data_NotSteg[:3331, 0:99] , Data_Steg[:960, 0:99], axis=0)
    Y_train = np.append(Data_NotSteg[:3331, -1] , Data_Steg[:960, -1], axis=0)

    X_test = np.append(Data_NotSteg[3332:, 0:99] , Data_Steg[961:, 0:99], axis=0)
    Y_test = np.append(Data_NotSteg[3332:, -1] , Data_Steg[961:, -1], axis=0)

    # relu, sigmoid
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=2000,
                                             batch_size=32,
                                             activation_function='sigmoid',
                                             dropout_p=0.2,
                                             verbose=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    scores.append(accuracy_score(Y_test, Y_pred))
    print('Accuracy: %f' % accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    print('*****************************************************')

print('All Accuracy Scores in ۳ Iteration: \n' + str(scores))
print('Mean Accuracy Scores in ۳ Iteration: ' + str(np.mean(scores)))