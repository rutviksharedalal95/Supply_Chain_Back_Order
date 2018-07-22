import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn import svm
from sklearn.metrics import auc, roc_curve
from imblearn import under_sampling as us
from imblearn import pipeline as pl

# create features
features_df = pd.read_csv("CleanedData.csv")
del features_df['went_on_backorder=Yes']

# Create outcomes
df = pd.read_csv("CleanedData.csv")
outcomes_df = df['went_on_backorder=Yes']

# Create X and y arrays
X = features_df.as_matrix()
y = outcomes_df.as_matrix()

# Apply the random under-sampling
pipeline = pl.make_pipeline(us.RandomUnderSampler(),
                            svm.SVC(kernel='rbf', C=1.0,probability=True)
                            )


# Split to test and train set
results=[]
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X,y):
    probas = pipeline.fit(X[train_index], y[train_index]).predict_proba(X[test_index])
    preds = probas[:, 1]
    fpr, tpr, threshold = roc_curve(y[test_index], preds)

    print("Show fpr  ----------------------------")
    print(fpr)
    print("Show tpr -----------------------------")
    print(tpr)

    np.savetxt('tpr.out', tpr)
    np.savetxt('fpr.out', fpr)

    roc_auc = auc(fpr, tpr)
    print("Show roc_auc   -----------------------")
    print(roc_auc)
    results.append(roc_auc)
   


print("AUC score is:  %.4f" % np.array(results).mean())
print("----------------")
