import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, PrecisionRecallDisplay

plt.rcParams["figure.figsize"] = [8, 6]

OneDCNN_True = np.load(
    r'E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\1D-CNN\ElasticNetCV-LassoCV\1D-CNN(Y_true)8.npy')
OneDCNN_Pro = np.load(
    r'E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\1D-CNN\ElasticNetCV-LassoCV\1D-CNN(Y_predict_probability)8.npy')

LSTM_True = np.load(
    r'E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\LSTM\ElasticNetCV-LassoCV\LSTM(Y_true)6.npy')
LSTM_Pro = np.load(
    r'E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\LSTM\ElasticNetCV-LassoCV\LSTM(Y_predict_probability)6.npy')
"""
KNN_True = np.load(
    r"D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\KNN-SVM\KNN-SVM(Y_true)5.npy")
KNN_Pro = np.load(
    r"D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\KNN-SVM\KNN-SVM(knn_Y_predict_probability)5.npy")

SVM_True = np.load(
    r"D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\KNN-SVM\KNN-SVM(Y_true)5.npy")
SVM_Pro = np.load(
    r"D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\KNN-SVM\KNN-SVM(svm_Y_predict_probability)5.npy")
"""
TwoDCNN_True = np.load(
    r"E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\2D-CNN\ElasticNetCV-LassoCV\2D-CNN(Y_true)5.npy")
TwoDCNN_Pro = np.load(
    r"E:\covid-coding\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\result\numpy data\2D-CNN\ElasticNetCV-LassoCV\2D-CNN(Y_predict_probability)5.npy")
'''
KNN_precision, KNN_recall, KNN_threshold = precision_recall_curve(KNN_True, KNN_Pro, pos_label=1)
SVM_precision, SVM_recall, SVM_threshold = precision_recall_curve(SVM_True, SVM_Pro, pos_label=1)
'''
OneDCNN_precision, OneDCNN_recall, OneDCNN_threshold = precision_recall_curve(OneDCNN_True, OneDCNN_Pro, pos_label=1)
LSTM_precision, LSTM_recall, LSTM_threshold = precision_recall_curve(LSTM_True, LSTM_Pro, pos_label=1)

"""
#2878b5
#9ac9db
#f8ac8c
#c82423
#ff8884
"""

_, ax = plt.subplots(figsize=(7, 8))


def plot_pr_multi_label(n_classes, Y_test, y_score):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # 绘制每个类的PR曲线和 iso-f1 曲线
    # setup plot details
    display = PrecisionRecallDisplay(recall=recall[1], precision=precision[1])
    display.plot(ax=ax, name=f"LassoCV-ElasticNetCV 2D-CNN", linestyle='--', color='#f8ac8c')

'''
plt.plot(KNN_recall, KNN_precision, color='#2878b5',
         lw=2, label='RandomForest KNN', linestyle='--')
plt.plot(SVM_recall, SVM_precision, color='#9ac9db',
         lw=2, label='LassoCV-ElasticNetCV SVM', linestyle='--')
'''
plt.plot(OneDCNN_recall, OneDCNN_precision, color='#c82423',
         lw=2, label='LassoCV-ElasticNetCV 1D-CNN', linestyle='--')
plot_pr_multi_label(2, TwoDCNN_True, TwoDCNN_Pro)
plt.plot(LSTM_recall, LSTM_precision, color='#ff8884',
         lw=2, label='LassoCV-ElasticNetCV LSTM', linestyle='--')

plt.plot([0, 1], [0, 1], lw=1.5, linestyle='-')
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('PR Curves', fontsize=15)
plt.legend(loc="best")
plt.show()
