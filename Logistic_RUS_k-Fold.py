import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from imblearn import under_sampling as us
from imblearn import pipeline as pl
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Create features
features_df = pd.read_csv("CleanedData.csv")
del features_df['went_on_backorder=Yes']

# Create outcomes
df = pd.read_csv("CleanedData.csv")
outcomes_df = df['went_on_backorder=Yes']


# Create X and y arrays
X = features_df.as_matrix()
y = outcomes_df.as_matrix()
print('Original dataset shape {}'.format(Counter(y)))


# Apply the random under-sampling
pipeline = pl.make_pipeline(us.RandomUnderSampler(),linear_model.LogisticRegression(
    penalty='l2',
    C=1,
    solver='liblinear',
    random_state=0)
    )


n_folds = 5
store_Roc_Auc = np.zeros(n_folds)
print(store_Roc_Auc)

# Split to test and train set
results=[]
skf = StratifiedKFold(n_splits=n_folds)
aaa = 0
for train_index, test_index in skf.split(X,y):
    probas = pipeline.fit(X[train_index], y[train_index]).predict_proba(X[test_index])
    preds = probas[:, 1]
    fpr, tpr, threshold = roc_curve(y[test_index], preds)

    print("Show fpr  ----------------------------")
    print(fpr)
    print("Show tpr -----------------------------")
    print(tpr)


    roc_auc = auc(fpr, tpr)

    results.append(roc_auc)
    print("Show roc_auc   -----------------------")
    print(roc_auc)
    
    store_Roc_Auc[aaa] = roc_auc
    print("Show store_Roc_Auc   -----------------------")
    print(store_Roc_Auc)
    aaa = aaa+1

print("Show average store_Roc_Auc   -----------------------")
print(np.mean(np.array(store_Roc_Auc)))

roc_auc = auc(fpr, tpr)




# Save ROC data
np.savetxt('tpr.out', tpr)
np.savetxt('fpr.out', fpr)



# # Plot of a ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Neural Network')
plt.legend(loc="lower right")
plt.show()





