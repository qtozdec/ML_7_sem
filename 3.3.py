import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

model = pd.read_csv("data-logistic.csv", header=None)
y = model[0]
X = model[[1, 2]]


def w1_search(X: pd.DataFrame, y: pd.Series, w1: float, w2: float, k: float, C: float):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[1][i] * (1.0 - 1.0 / (1.0 + np.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i]))))

    return w1 + (k * (1.0 / l) * S) - k * C * w1


def w2_search(X: pd.DataFrame, y: pd.Series, w1: float, w2: float, k: float, C: float):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[2][i] * (1.0 - 1.0 / (1.0 + np.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i]))))

    return w2 + (k * (1.0 / l) * S) - k * C * w2


def gradient_descent(X: pd.DataFrame, y: pd.Series, ):
    w1 = 0
    w2 = 0
    for i in range(10000):
        temp1, temp2 = w1, w2
        w1, w2 = w1_search(X, y, w1, w2, 0.1, 0), w2_search(X, y, w1, w2, 0.1, 0)
        if np.sqrt((temp1 - w1) ** 2 + (temp2 - w2) ** 2) <= 1e-5:
            break

    return w1, w2


def gradient_descent_2(X: pd.DataFrame, y: pd.Series):
    w1 = 0
    w2 = 0
    for i in range(10000):
        temp1, temp2 = w1, w2
        w1, w2 = w1_search(X, y, w1, w2, 0.1, 10), w2_search(X, y, w1, w2, 0.1, 10)
        if np.sqrt((temp1 - w1) ** 2 + (temp2 - w2) ** 2) <= 1e-5:
            break

    return w1, w2


w1, w2 = gradient_descent(X, y)
w1_reg, w2_reg = gradient_descent_2(X, y)


def ax(X: pd.DataFrame, w1: float, w2: float):
    return 1.0 / (1.0 + np.exp(-w1 * X[1] - w2 * X[2]))


auc = roc_auc_score(y, ax(X, w1, w2))
auc_reg = roc_auc_score(y, ax(X, w1_reg, w2_reg))

print(f"Без регуляции: {auc:.3f}")
print(f"С регуляцией: {auc_reg:.3f}")
