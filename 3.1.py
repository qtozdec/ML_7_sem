import pandas
from sklearn.svm import SVC

# Загружаем выборку, определяем признаки и целевой вектор
data = pandas.read_csv('svm-data.csv', header=None, usecols=[0])
target = pandas.read_csv('svm-data.csv', header=None, usecols=[1, 2])
X = target.to_numpy()
y = data[0].to_numpy()
# Обучаем классификатор
clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(X, y)
print('Опорные объекты:')
for i in clf.support_:
    print(i + 1)
