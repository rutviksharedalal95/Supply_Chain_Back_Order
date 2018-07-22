from sklearn.neural_network import MLPClassifier
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import auc, roc_curve

#import matplotlib.pyplot as plt
# Create features
features_df = pd.read_csv("DataRUS.csv")
del features_df['went_on_backorder.Yes'] #del = delete column

# Create outcomes
df = pd.read_csv("DataRUS.csv")
outcomes_df = df['went_on_backorder.Yes']



#-------------------------------------------------
# Create X and y arrays
X = features_df.as_matrix()
y = outcomes_df.as_matrix()

# Split the data set in a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Setup classifier
model = MLPClassifier()

# Setup grid search parameters
para_grid = {
    'activation': ['tanh'],
    'hidden_layer_sizes': [(30,200,200,30),(100,50,25,5)],
    'learning_rate': ['constant']
}

grid_cv = GridSearchCV(model, para_grid, n_jobs=2)
grid_cv.fit(X_train, y_train)

print("Best parameters are: ")
print(grid_cv.best_params_)
print("---------------------")
print(type(grid_cv.best_params_))

with open('NN_best_params.txt', 'w') as file:
    file.write(json.dumps(grid_cv.best_params_))
# np.savetxt('NN_best_params.txt', grid_cv.best_params_)

# Find accuracy of test set
probs = grid_cv.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, _ = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

print("AUC score is:  %.4f" % roc_auc)
print("----------------")


# Save ROC data
np.savetxt('tpr.out', tpr)
np.savetxt('fpr.out', fpr)
