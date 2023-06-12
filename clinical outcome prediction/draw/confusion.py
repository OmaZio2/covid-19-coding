"""
临床结局部分
    绘制 特征数量最少且准确率最高的分类组合
        KNN:
        SVM:
        TA 1D-CNN:
        TA LSTM:
        2D CNN:
"""
import numpy as np
import matplotlib.pyplot as plt
plt.figure(1)
classes = ['0','1']

"""
TA LSTM: [[193   2]
            [  0  15]]       对了
TA 1D-CNN:  [[191   4]
                [  1  14]]
2D-CNN: [[194   1]
         [  2  13]]       对了
"""
confusion_matrix = np.array([[193 ,  2],
                                [ 3 , 12]])
confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
confusion_matrix= np.around(confusion_matrix, 4)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes,fontsize=22)
plt.yticks(tick_marks, classes,fontsize=22)
thresh = confusion_matrix.max() / 2.
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(2)] for i in range(2)],(confusion_matrix.size,2))
for i, j in iters:
    plt.text(j-0.27, i+0.07, format(confusion_matrix[i, j]),fontsize=22)   #显示对应的数字
plt.ylabel('Real label',fontsize=22,labelpad=-5)
plt.xlabel('Prediction',fontsize=22,labelpad=-5)
plt.tight_layout()
plt.savefig("../result/数据可视化/KNN-confusion-clinical_outcome.png")
plt.show()