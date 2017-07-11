import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans

train = pd.read_csv('water-treatment.data.txt', header=None, na_values=["?"],
                    names=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14',
                           'G15', 'G16', 'G17', 'G18', 'G19', 'G20', 'G21', 'G22', 'G23', 'G24', 'G25', 'G26', 'G27',
                           'G28', 'G29', 'G30', 'G31', 'G32', 'G33', 'G34', 'G35', 'G36', 'G37', 'G38', 'G39'])

print(train.shape)
print(train.describe())
print(train.head())


def num_missing(x):
    return sum(x.isnull())


print(train.apply(num_missing, axis=0))

train = train.apply(lambda x: x.fillna(x.value_counts().index[0]))

train = train.drop('G1', axis=1)

for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(train)
    print("k =", k)
    print(sklearn.metrics.silhouette_score(train, kmeans.labels_))

    # print(kmeans.labels_)
