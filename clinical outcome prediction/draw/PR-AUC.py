import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay



plt.rcParams["figure.figsize"] = [8, 6]

KNN_True = np.load("../result/numpy data/KNN/ElasticNetCV-LassoCV/KNN(Y_true)7.npy")
KNN_Pro = np.load("../result/numpy data/KNN/ElasticNetCV-LassoCV/KNN(Y_predict_probability)7.npy")

TAOneDCNN_True = np.load("../result/numpy data/TA_1DCNN/2 TA_1D-CNN(Y_true)5.npy")
TAOneDCNN_Pro = np.load("../result/numpy data/TA_1DCNN/2 TA_1D-CNN(Y_predict_probability)5.npy")


TALSTM_True = np.load("../result/numpy data/TA_LSTM/2 TA_LSTM(Y_true)5.npy")
TALSTM_Pro = np.load("../result/numpy data/TA_LSTM/2 TA_LSTM(Y_predict_probability)5.npy")

TwoDCNN_True = np.load("../result/numpy data/2D-CNN/2 2D-CNN(Y_true)9.npy")
TwoDCNN_Pro = np.load("../result/numpy data/2D-CNN/2 2D-CNN(Y_predict_probability)9.npy")

KNN_precision, KNN_recall, KNN_threshold =  precision_recall_curve(KNN_True, KNN_Pro,pos_label=1)
KNN_AUC = average_precision_score(KNN_True, KNN_Pro)

TAOneDCNN_precision, TAOneDCNN_recall, TAOneDCNN_threshold =  precision_recall_curve(TAOneDCNN_True, TAOneDCNN_Pro,pos_label=1)
TAOneDCNN_AUC = average_precision_score(TAOneDCNN_True, TAOneDCNN_Pro, pos_label=1)

TALSTM_precision, TALSTM_recall, TALSTM_threshold =  precision_recall_curve(TALSTM_True, TALSTM_Pro,pos_label=1)
TALSTM_AUC = average_precision_score(TALSTM_True, TALSTM_Pro,  pos_label=1)



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
    _, ax = plt.subplots(figsize=(7, 8))
    display = PrecisionRecallDisplay(recall=recall[1], precision=precision[1])
    display.plot(ax=ax, name=f"ElasticNetCV_LassoCV 2D-CNN",lw=2,linestyle='-', color='#f8ac8c')



plot_pr_multi_label(2,TwoDCNN_True,TwoDCNN_Pro)



"""
#2878b5
#9ac9db
#f8ac8c
#c82423
#ff8884
"""
plt.xlim(0.5,1.05)
plt.ylim(0.5,1.05)

plt.plot(KNN_recall,KNN_precision, color='#2878b5',
         lw=2, label='ElasticNetCV KNN', linestyle='-')
plt.plot(TAOneDCNN_recall,TAOneDCNN_precision,color='#c82423',alpha = 0.8,
         lw=2, label='ElasticCV_LassoCV TA 1D-CNN', linestyle='-')
plt.plot(TALSTM_recall,TALSTM_precision, color='#00FF00',alpha = 0.6,
         lw=2, label='ElasticNetCV_LassoCV TA LSTM', linestyle='-')


plt.plot([0.5, 1], [0.5, 1], lw=1.5, linestyle='--')
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('PR Curves',fontsize=15)
plt.legend(loc="best")
plt.savefig("../result/数据可视化/PR-AUC-clinical_outcome.png")
plt.show()
