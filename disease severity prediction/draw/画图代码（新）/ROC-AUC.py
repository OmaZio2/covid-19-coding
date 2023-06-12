import matplotlib.pyplot as plt

import numpy as np
from numpy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

# svm_roc_auc= auc(svm_fpr, svm_tpr)  ###计算auc的值，auc就是曲线包围的面积，越大越好
plt.rcParams["figure.figsize"] = [8, 6]

OneDCNN_True = np.load(
    r'E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\1D-CNN\ElasticNetCV-LassoCV\1D-CNN(Y_true)8.npy')
OneDCNN_Pro = np.load(
    r'E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\1D-CNN\ElasticNetCV-LassoCV\1D-CNN(Y_predict_probability)8.npy')

LSTM_True = np.load(
    r'E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\LSTM\ElasticNetCV-LassoCV\LSTM(Y_true)6.npy')
LSTM_Pro = np.load(
    r'E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\LSTM\ElasticNetCV-LassoCV\LSTM(Y_predict_probability)6.npy')
'''
KNN_True = np.load(
    r"D:\pythonProject\COVID-19-Prediction\disease severity prediction\draw\PR-ROC data\RandomForest KNN(5)正确标签.npy")
KNN_Pro = np.load(
    r"D:\pythonProject\COVID-19-Prediction\disease severity prediction\draw\PR-ROC data\RandomForest KNN(5)概率输出值.npy")

SVM_True = np.load(
    r"D:\pythonProject\COVID-19-Prediction\disease severity prediction\draw\PR-ROC data\LassoCV-ElasticNetCV SVM(9)正确标签.npy")
SVM_Pro = np.load(
    r"D:\pythonProject\COVID-19-Prediction\disease severity prediction\draw\PR-ROC data\LassoCV-ElasticNetCV SVM(9)概率输出值.npy")
'''
TwoDCNN_True = np.load(
    r"E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\2D-CNN\ElasticNetCV-LassoCV\2D-CNN(Y_true)5.npy")
TwoDCNN_Pro = np.load(
    r"E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\2D-CNN\ElasticNetCV-LassoCV\2D-CNN(Y_predict_probability)5.npy")
'''
KNN_fpr, KNN_tpr, KNN_threshold = roc_curve(KNN_True, KNN_Pro)
KNN_AUC = auc(KNN_fpr, KNN_tpr)

SVM_fpr, SVM_tpr, SVM_threshold = roc_curve(SVM_True, SVM_Pro)
SVM_AUC = auc(SVM_fpr, SVM_tpr)
'''
OneCNN_fpr, OneCNN_tpr, OneCNN_threshold = roc_curve(OneDCNN_True, OneDCNN_Pro)
OneCNN_AUC = auc(OneCNN_fpr, OneCNN_tpr)

LSTM_fpr, LSTM_tpr, TALSTM_threshold = roc_curve(LSTM_True, LSTM_Pro)
LSTM_AUC = auc(LSTM_fpr, LSTM_tpr)

"""
#2878b5
#9ac9db
#f8ac8c
#c82423
#ff8884
"""

"""
2D-CNN 比较奇葩
"""
_, ax = plt.subplots(figsize=(7, 8))
'''
plt.plot(KNN_fpr, KNN_tpr, color='#2878b5',
         lw=2, label='RandomForest KNN (AUC:%0.5f)' % KNN_AUC, linestyle='--')
plt.plot(SVM_fpr, SVM_tpr, color='#9ac9db',
         lw=2, label='LassoCV-ElasticNetCV SVM (AUC:%0.5f)' % SVM_AUC, linestyle='--')
'''
plt.plot(OneCNN_fpr, OneCNN_tpr, color='#c82423',
         lw=2, label='LassoCV-ElasticNetCV 1D-CNN (AUC:%0.5f)' % OneCNN_AUC, linestyle='--')

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(TwoDCNN_True[:, i], TwoDCNN_Pro[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    # macro（方法一）
    # First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw = 2
plt.plot(fpr["macro"], tpr["macro"],
         label='LassoCV-ElasticNetCV 2D-CNN (AUC:{0:0.5f})'
               ''.format(roc_auc["macro"]),
         color='#f8ac8c', linestyle='-', linewidth=2)

plt.plot(LSTM_fpr, LSTM_tpr, color='#ff8884',
         lw=2, label='LassoCV-ElasticNetCV LSTM (AUC:%0.5f)' % LSTM_AUC, linestyle='--')

plt.plot([0, 1], [0, 1], lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('ROC Curves', fontsize=15)
plt.legend(loc="best")
plt.show()
