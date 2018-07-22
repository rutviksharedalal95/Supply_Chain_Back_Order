import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc
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

# Setup classifier
model = ensemble.GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=10,
    max_depth=9,
    min_samples_leaf=5,
    max_features=1,
    loss='deviance'
)

results = []
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X,y):
    probas = model.fit(X[train_index],y[train_index]).predict_proba(X[test_index])
    preds = probas[:,1]
    fpr,tpr,threshold = roc_curve(y[test_index],preds)
    roc_auc = auc(fpr,tpr)
    results.append(roc_auc)

print("AUC score is: %.4f" % np.array(results).mean())
print("--------------")

# Plot of a ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.legend(loc="lower right")
plt.show()

# Save ROC data
np.savetxt('tpr.out', tpr)
np.savetxt('fpr.out', fpr)