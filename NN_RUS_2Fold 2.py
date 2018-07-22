from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


#Apply the Random Under-sampling
pipeline = pl.make_pipeline(us.RandomUnderSampler(),
                                MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto',
                                 beta_1=0.9, beta_2=0.999, early_stopping=False,
                                 epsilon=1e-08, hidden_layer_sizes=(100,35,15,100), learning_rate='constant',
                                 learning_rate_init=0.001, max_iter=200,
                                  random_state=1, shuffle=True,
                                 solver='adam', tol=0.0001, verbose=False,
                                 warm_start=False)
                            )


# split to test and train set
results = []
skf = StratifiedKFold(n_splits=2)
for train_index, test_index in skf.split(X,y):
    probas = pipeline.fit(X[train_index],y[train_index]).predict_proba(X[test_index])
    preds = probas[:,1]
    fpr,tpr,threshold = roc_curve(y[test_index],preds)
    roc_auc = auc(fpr,tpr)
    results.append(roc_auc)

print("AUC score is: %.4f" % np.array(results).mean())
print("--------------")


# Save ROC data
np.savetxt('tpr.out', tpr)
np.savetxt('fpr.out', fpr)


#
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


