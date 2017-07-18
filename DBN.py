import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from dbn.tensorflow import SupervisedDBNClassification
# from dbn import SupervisedDBNClassification


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

Data_NotSteg = Z[Z[:,100] == 0];
Data_Steg = Z[Z[:,100] == 1][0:1200,:]

Data = np.append(Data_NotSteg , Data_Steg , axis=0)
X = Data[:,0:AttributeCount]
Y = Data[:,-1]

for i in range(3):
    print('Iteration ' + str(i+1))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
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