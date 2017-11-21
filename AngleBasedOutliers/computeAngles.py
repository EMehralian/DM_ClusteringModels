import pandas as pd
import numpy as np
from tqdm import tqdm
from Angle import *


data = pd.read_csv('Data/u2r.csv',header=None,sep=';')

print(data.shape)

for i in tqdm(enumerate(data)):
    for j in tqdm(enumerate(data)):
        for k in tqdm(enumerate(data)):
            print(toDegrees(angle(i, j, k)))

            
