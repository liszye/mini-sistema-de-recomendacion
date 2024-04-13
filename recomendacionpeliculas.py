import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import numpy as np

# Cargar datos (ejemplo)
data = {
    'Usuario': ['Usuario1', 'Usuario2', 'Usuario3', 'Usuario4'],
    'Avatar (2009)': [5, 4, 0, 0],
    'Vengadores:Endgame(2019)': [0, 0, 3, 4],
    'Avatar: El sentido del agua (2022)': [2, 0, 0, 5],
    'Titanic (1997)': [0, 2, 4, 0]
}
df = pd.DataFrame(data)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


knn_model = NearestNeighbors(n_neighbors=2)  
knn_model.fit(train_data.iloc[:, 1:])  


distances, indices = knn_model.kneighbors(test_data.iloc[:, 1:])


y_true = test_data.iloc[:, 1:].values.flatten()
y_pred = np.where(train_data.iloc[indices, 1:].values.flatten() > 0, 1, 0)


