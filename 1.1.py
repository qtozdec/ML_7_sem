import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("train.csv", index_col="PassengerId")

sex_counts = data["Sex"].value_counts()
print(f"Мужчин: {sex_counts['male']} Женщин: {sex_counts['female']}")

survived_counts = data["Survived"].value_counts()
survived_percent = 100.0 * survived_counts[1] / survived_counts.sum()
print(f"Доля выживших: {survived_percent:.2f}")

pclass_counts = data["Pclass"].value_counts()
pclass_percent = 100.0 * pclass_counts[1] / pclass_counts.sum()
print(f"Доля пассажиров первого класса{pclass_percent:.2f}")

ages = data["Age"].dropna()
print(f"Средний возраст: {ages.mean():.2f} Медиана: {ages.median():.2f}")

corr = data["SibSp"].corr(data["Parch"])
print(f"Корреляция Пирсона: {corr:.2f}")

# Вычисление важности признаков
data = data[np.isnan(data['Age']) == False]
X = data.filter(items=['Sex', 'Pclass', 'Fare', 'Age']).replace({'Sex': {'male': 1, 'female': 0}})
y = data.filter(items=['Survived'])
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(" ".join(feature_importances.head(2).index))


