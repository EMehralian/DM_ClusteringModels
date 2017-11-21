import pandas as pd
import time
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import v_measure_score

train = pd.read_csv('./HTRU2/HTRU_2.csv', header=None , names=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'])

print(train.head())

model = AgglomerativeClustering(2, linkage='ward')
model.fit(train)

Y = train['G8']
X = train.drop('G8', axis=1)


for index, linkage in enumerate(('complete', 'ward')):
    plt.subplot(1, 3, index + 1)
    model = AgglomerativeClustering(linkage=linkage, n_clusters=2)
    predict = model.fit(X)
    print("NMI",normalized_mutual_info_score(predict.labels_, Y))

    print("v measure",v_measure_score(Y, predict.labels_))