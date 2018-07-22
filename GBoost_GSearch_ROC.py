import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt (Python 2.7 doesn't have MatplotLib)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc
from imblearn import pipeline as pl
from imblearn import under_sampling as us



# Create features
features_df = pd.read_csv("CleanedData.csv")
del features_df['went_on_backorder=Yes']

# Create outcomes
df = pd.read_csv("CleanedData.csv")
outcomes_df = df['went_on_backorder=Yes']

# Create X and y arrays
X = features_df.as_matrix()
y = outcomes_df.as_matrix()

# Split the data set in a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Setup classifier
model = ensemble.GradientBoostingClassifier()

# Setup grid search parameters
para_grid = {
    'learning_rate': [0.05,0.1],
    'n_estimators': [10,20,100,200,500],
    'max_depth': [3, 6,9,10,12],
    'min_samples_leaf': [3,6,9,10,12],
    'max_features': [0.1,1,10],
    'loss': ['deviance'],
    'random_state': [0]
}
grid_cv = GridSearchCV(model, para_grid, n_jobs=8 )

# Train you model
grid_cv.fit(X_train, y_train)
print("Our model is created.")
print("Best parameters are: ")
print(grid_cv.best_params_)
print("---------------------")

# Find accuracy of test set
y_prob = grid_cv.predict_proba(X_test)
y_pred = y_prob[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("AUC score is:  %.4f" % roc_auc)
print("---------------------")

# Save ROC data
np.savetxt('tpr.out', tpr)
np.savetxt('fpr.out', fpr)
