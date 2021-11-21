from sklearn import datasets
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# Загружаем объекты из датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм"
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
vectorizer = TfidfVectorizer()
# Определяем признаки TF-IDF и целевой вектор
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
# Подбираем минимальный лучший параметр C
grid = {"C": np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel="linear", random_state=241)
gs = GridSearchCV(clf, grid, scoring="accuracy", cv=cv, n_jobs=-1)
gs.fit(X, y)
C = gs.best_params_.get('C')
# Обучаем модель с нашим параметром С
clf = svm.SVC(C=C, kernel="linear", random_state=241)
clf.fit(X, y)
# Находим слова
words = np.array(vectorizer.get_feature_names())
word_array = pd.Series(clf.coef_.data, index=words[clf.coef_.indices])
top_words = word_array.abs().sort_values(ascending=False)[:10]
print(top_words)
