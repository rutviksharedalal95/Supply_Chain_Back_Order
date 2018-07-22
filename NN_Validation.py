from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import auc, roc_curve

from sklearn import preprocessing
from imblearn import under_sampling as us
from imblearn import pipeline as pl

# Create features
features_df = pd.read_csv("CleanedData.csv")
del features_df['went_on_backorder=Yes'] #del = delete column

# Create outcomes
df = pd.read_csv("CleanedData.csv")
outcomes_df = df['went_on_backorder=Yes']



#-------------------------------------------------
# Create X and y arrays
X = features_df.as_matrix()
y = outcomes_df.as_matrix()


# Setup classifier
model = MLPClassifier(activation='tanh', hidden_layer_sizes=(100, 50,25,5), learning_rate='constant'
)


# split to test and train set
results = []
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X,y):
    probas = model.fit(X[train_index],y[train_index]).predict_proba(X[test_index])
    preds = probas[:,1]
    fpr,tpr,threshold = roc_curve(y[test_index],preds)
    roc_auc = auc(fpr,tpr)
    results.append(roc_auc)
    print("the ROC score in 5-fold validation: %.4f" % roc_auc)

print("AUC score is: %.4f" % (np.array(results)).mean())
print("--------------")

