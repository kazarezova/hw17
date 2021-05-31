import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

df = pd.read_csv("bc_data.csv", index_col=0)
ann = pd.read_csv("bc_ann.csv", index_col=0)
X_train = df.loc[ann.loc[ann["Dataset type"] == "Training"].index]
X_test = df.loc[ann.loc[ann["Dataset type"] == "Validation"].index]

ttest=[ttest_ind(X_train[gene], X_test[gene])[1] for gene in df.columns]
count=0
for i in ttest:
    if i<0.05:
        count+=1
print(count/len(X_test.columns))

# 0.6004895645452203
