import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
classes = ['0', '1']
'''
症状程度
 
LSTM   [[81 18]
 [26 85]]
         
1D-CNN [[76 23]
 [22 89]]
         
2D-CNN  [[67 32]
 [14 97]]
KNN:    [[392 ,124]],[
         [159 ,373]]
SVM   [[386, 130],[
        122 ,410]]
         
Severity-Confusion-LSTM
'''

confusion_matrix = np.array([[81, 18],
 [26, 85]])
confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
confusion_matrix = np.around(confusion_matrix, 4)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=22)
plt.yticks(tick_marks, classes, fontsize=22)
thresh = confusion_matrix.max() / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# ij配对，遍历矩阵迭代器
iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (confusion_matrix.size, 2))
for i, j in iters:
    plt.text(j - 0.27, i + 0.07, format(confusion_matrix[i, j]), fontsize=22)  # 显示对应的数字
plt.ylabel('Real label', fontsize=22, labelpad=-5)
plt.xlabel('Prediction', fontsize=22, labelpad=-5)
plt.tight_layout()
plt.show()
