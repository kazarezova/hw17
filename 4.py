import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, plot_roc_curve

import matplotlib.pyplot as plt

df = pd.read_csv("bc_data.csv", index_col=0)
ann = pd.read_csv("bc_ann.csv", index_col=0)
genes = "TRIP13;UBE2C;ZWINT;EPN3;KIF4A;ECHDC2;MTFR1;STARD13;IGFBP6;NUMA1;CCNL2".split(";")

X_train = df.loc[ann.loc[ann["Dataset type"] == "Training"].index].to_numpy()
y_train = ann.loc[ann["Dataset type"] == "Training", "Class"].to_numpy()

X_test = df.loc[ann.loc[ann["Dataset type"] == "Validation"].index].to_numpy()
y_test = ann.loc[ann["Dataset type"] == "Validation", "Class"].to_numpy()

model = SVC(kernel="linear")
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print(accuracy_score(y_train, y_pred))
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

M = confusion_matrix(y_test, y_pred)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)
plot_roc_curve(model, X_test, y_test)
plt.plot(1 - TPR, TNR, "x", c="red")
plt.savefig("bc_all.png", dpi=300)

df = df[genes]
X_train = df.loc[ann.loc[ann["Dataset type"] == "Training"].index].to_numpy()
y_train = ann.loc[ann["Dataset type"] == "Training", "Class"].to_numpy()

'''
1.0
0.6770833333333334
0.75 0.4583333333333333
0.7467532467532467
0.7395833333333334
0.75 0.7083333333333334

X_test = df.loc[ann.loc[ann["Dataset type"] == "Validation"].index].to_numpy()
y_test = ann.loc[ann["Dataset type"] == "Validation", "Class"].to_numpy()

model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print(accuracy_score(y_train, y_pred))
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

M = confusion_matrix(y_test, y_pred)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)
plot_roc_curve(model, X_test, y_test)
plt.plot(1 - TPR, TNR, "x", c="red")
plt.savefig("bc_11.png", dpi=300)
