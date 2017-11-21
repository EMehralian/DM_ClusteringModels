import scipy.io
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import pandas as pd
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

data = scipy.io.loadmat("cardio.mat")

for i in data:
    if '__' not in i and 'readme' not in i:
        np.savetxt(("cardio" + i + ".csv"), data[i], delimiter=',')

xTrain = pd.read_csv("cardioX.csv")
yTrain = pd.read_csv("cardioY.csv")

# print(xTrain.head())
e = EmpiricalCovariance().fit(xTrain)

# print(__doc__)
#
# rng = np.random.RandomState(42)
#
# # define two outlier detection tools to be compared
# classifiers = {
#     "One-Class SVM": svm.OneClassSVM(nu=0.95 * 0.25 + 0.05,
#                                      kernel="rbf", gamma=0.1),
#     "Isolation Forest": IsolationForest(max_samples=80,
#                                         contamination=0.25,
#                                         random_state=rng)
# }
#
# # Fit the problem with varying cluster separation
# X = xTrain
# print(X.shape)
#
# # Fit the model
# for i, (clf_name, clf) in enumerate(classifiers.items()):
#     clf.fit(X)
#     y_pred = clf.predict(X)
#     print(y_pred)
#
# t = 0
# for i in range(0, len(y_pred)):
#     if (y_pred[i] == 1):
#         t = t + 1
#
# print(t)
