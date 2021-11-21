from sklearn.model_selection import ( KFold,  cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import numpy as np
import pandas

WineClass = pandas.read_csv('wine.data', header=None, usecols=[0])
WineParams = pandas.read_csv('wine.data', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
n_splits = 5
X = WineParams.to_numpy()
y = WineClass[0].to_numpy()
cv = KFold(n_splits, shuffle=True, random_state=42)

result = []
# Находим точность классификации на кросс-валидации для метода k ближайших соседей при k = 1-50
for i in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X, y)
    result.append(np.average(cross_val_score(neigh, X, y, cv=cv)))
score = result[np.argmax(result)]
print('До масштабирования')
print(f"При k = {np.argmax(result)+1}")
print(f"{score:.2f}")

result = []
# масштабирование признаков
k = scale(X)
# Находим точность классификации на кросс-валидации для метода k ближайших соседей при k = 1-50 при масштабировании

for i in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(k, y)
    result.append(np.average(cross_val_score(neigh, k, y, cv=cv)))
print('После масштабирования')
print(f"При k = {np.argmax(result)+1}")
score = result[np.argmax(result)]
print(f"{score:.2f}")
