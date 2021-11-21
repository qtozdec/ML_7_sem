import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve

model = pd.read_csv("classification.csv")
# Считаем величины TP, FP, FN и TN
TP = len(model[(model["pred"] == 1) & (model["true"] == 1)])
FP = len(model[(model["pred"] == 1) & (model["true"] == 0)])
FN = len(model[(model["pred"] == 0) & (model["true"] == 1)])
TN = len(model[(model["pred"] == 0) & (model["true"] == 0)])
# print(f"{len(TP)} {len(FP)} {len(FN)} {len(TN)}")
print(f"Величины TP, FP, FN и TN равны {TP}, {FP}, {FN}, {TN} ")

# Считаем основные метрики качества классификатора
accuracy = accuracy_score(model["true"], model["pred"])
precision = precision_score(model["true"], model["pred"])
recall = recall_score(model["true"], model["pred"])
f_mera = f1_score(model["true"], model["pred"])
print(f"Основные метрики качества классификатора: \n accuracy: {accuracy:.2f} ,precision: {precision:.2f}, recall: {recall:.2f}, f_mera: {f_mera:.2f}")

model2 = pd.read_csv("scores.csv")
names = model2.columns[1:]
# Определяем, какой классификатор имеет наибольшее значение метрики AUC-ROC
scores = pd.Series([roc_auc_score(model2["true"], model2[clf]) for clf in names], index=names)
print(f"Классификатор, который имеет наибольшее значение метрики AUC-ROC: {scores.sort_values(ascending=False).index[0]}")

# Определяем, какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70%
precision = []
for clf in names:
    pr_curve = precision_recall_curve(model2["true"], model2[clf])
    precision.append(pr_curve[0][pr_curve[1] >= 0.7].max())

print(f"Классификатор, который достигает наибольшей точности (Precision) при полноте (Recall) не менее 70%: {names[np.argmax(precision)]}")
