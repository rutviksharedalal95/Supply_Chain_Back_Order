import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Import some data to play with
tpr_NN = np.loadtxt('tpr_NN.out')
fpr_NN = np.loadtxt('fpr_NN.out')
roc_auc_NN = auc(fpr_NN, tpr_NN)

tpr_GB = np.loadtxt('tpr_GB.out')
fpr_GB = np.loadtxt('fpr_GB.out')
roc_auc_GB = auc(fpr_GB, tpr_GB)

tpr_SVM = np.loadtxt('tpr_SVM_5.out')
fpr_SVM = np.loadtxt('fpr_SVM_5.out')
roc_auc_SVM = auc(fpr_SVM, tpr_SVM)

tpr_GB_noGS = np.loadtxt('tpr_GB_noGS.out')
fpr_GB_noGS = np.loadtxt('fpr_GB_noGS.out')
roc_auc_GB_noGS = auc(fpr_GB_noGS, tpr_GB_noGS)


tpr_NN_noGS = np.loadtxt('tpr_NN_noGS.out')
fpr_NN_noGS = np.loadtxt('fpr_NN_noGS.out')
roc_auc_NN_noGS = auc(fpr_NN_noGS, tpr_NN_noGS)


tpr_Logistic_RUS = np.loadtxt('tpr_Logistic_RUS.out')
fpr_Logistic_RUS = np.loadtxt('fpr_Logistic_RUS.out')
roc_auc_Logistic_RUS = auc(fpr_Logistic_RUS, tpr_Logistic_RUS)

tpr_Logistic_SMOTE = np.loadtxt('tpr_Logistic_SMOTE.out')
fpr_Logistic_SMOTE = np.loadtxt('fpr_Logistic_SMOTE.out')
roc_auc_Logistic_SMOTE = auc(fpr_Logistic_SMOTE, tpr_Logistic_SMOTE)


# Plot of a ROC curve for a specific class
# plt.figure()
plt.plot(fpr_NN, tpr_NN, color='darkorange',
         lw=2, label='NN + Grid Search (area = %0.2f)' % roc_auc_NN)
plt.plot(fpr_GB, tpr_GB, color='red',
         lw=2, label='GB + Grid Search (area = %0.2f)' % roc_auc_GB)
plt.plot(fpr_SVM, tpr_SVM, color='green',
         lw=2, label='SVM (area = %0.2f)' % roc_auc_SVM)
plt.plot(fpr_GB_noGS, tpr_GB_noGS, color='pink',
         lw=2, label='GB (area = %0.2f)' % roc_auc_GB_noGS)
plt.plot(fpr_NN_noGS, tpr_NN_noGS, color='blue',
         lw=2, label='NN (area = %0.2f)' % roc_auc_NN_noGS)
plt.plot(fpr_Logistic_RUS, tpr_Logistic_RUS, color='purple',
         lw=2, label='Logistic + RUS (area = %0.2f)' % roc_auc_Logistic_RUS)
plt.plot(fpr_Logistic_SMOTE, tpr_Logistic_SMOTE, color='yellow',
         lw=2, label='Logistic + SMOTE (area = %0.2f)' % roc_auc_Logistic_SMOTE)
