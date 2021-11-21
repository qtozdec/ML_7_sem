import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Загружаем выборку, определяем признаки и целевой вектор
data = pandas.read_csv('perceptron-train.csv', header=None, usecols=[0])
target = pandas.read_csv('perceptron-train.csv', header=None, usecols=[1, 2])
data_test = pandas.read_csv('perceptron-test.csv', header=None, usecols=[0])
target_test = pandas.read_csv('perceptron-test.csv', header=None, usecols=[1, 2])

X_train = target.to_numpy()
y_train = data[0].to_numpy()
X_test = target_test.to_numpy()
y_test = data_test[0].to_numpy()

# Обучение персептрона обучающей выборкой
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

# Вычисление качества тестовой выборки
acc_before = accuracy_score(y_test, clf.predict(X_test))

# Нормализация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение персептрона обучающей выборкой после нормализации
clf.fit(X_train_scaled, y_train)
# Вычисление качества тестовой выборки
acc_after = accuracy_score(y_test, clf.predict(X_test_scaled))

print(f"Качество до нормализации тестовой выборки - {acc_after},"
      f" качество после нормализации - {acc_before}, разница составила"
      f" {round(acc_after - acc_before, 3)}")
