import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import math

data_url = "./mammadAgha.csv"
data = pd.read_csv(data_url).dropna()

trainSize = math.ceil(data.shape[0]*0.8)
data = data.sample(frac=1)
train = data[:trainSize]
test = data[trainSize:]

y_train = train['ABOF']
x_train = train.drop("ABOF", axis=1)

y_test = test['ABOF']
x_test = test.drop("ABOF", axis=1)

LR1 = LogisticRegression(penalty='l1', tol=0.01)
LR2 = LogisticRegression(penalty='l2', tol=0.01)
DT = DecisionTreeClassifier(random_state=0, max_depth=15, min_samples_leaf=2)
RF = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1,verbose=True)
NN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(600, 300), random_state=1, activation='relu', verbose=True, max_iter=20)


leanerSVML1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                        random_state=0)
leanerSVML2 = LinearSVC(penalty='l2', loss='hinge', dual=True, random_state=0)

clf = svm.SVC(probability=True, verbose=True)

kf = KFold(n_splits=10, random_state=None, shuffle=False)

X = x_train.values
y = y_train.values

def classifing(classifier):
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        classifier.fit(X_train, Y_train)
        print("clf fitted")
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print("Fscore", f1_score(y_test, prediction))
    print("accuracy", accuracy_score(y_test, prediction))
    print(recall_score(y_test, prediction, average=None))
    if classifier != leanerSVML1 and classifier != leanerSVML2:
        probPrediction = classifier.predict_proba(x_test)
        print("log loss", log_loss(y_test, probPrediction))
    return prediction



outlierList = classifing(DT)
print(outlierList)
