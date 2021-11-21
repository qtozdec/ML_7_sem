from sklearn.model_selection import (KFold, cross_val_score)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
import numpy as np
from sklearn.datasets import load_boston

n_splits = 5
cv = KFold(n_splits, shuffle=True, random_state=42)
# Загружаем выборку, определяем признаки и целевой вектор
X, y = load_boston(return_X_y=True)
# Производим масштабирование
X_scale = scale(X)
result = []
array = np.linspace(1, 10, 200)
# Находим оптимальное р при тесте двухсот вариантов
for p in array:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    result.append(np.average(cross_val_score(neigh, X_scale, y, cv=cv, scoring='neg_mean_squared_error')))

i = np.argmax(result)
print(f"Оптимальное решение при p = {i+1}")
