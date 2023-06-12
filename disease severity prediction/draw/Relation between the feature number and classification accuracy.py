import matplotlib.pyplot as plt

x_10 = [10, 9, 8, 7, 6, 5]

LassoCVElasticNetCVLSTM = [0.75859, 0.76718, 0.75286, 0.74809, 0.74427, 0.74237]
RandomForestLSTM = [0.74523, 0.75, 0.75095, 0.75095, 0.74523, 0.74427]

LassoCVElasticNetCVOneCNN = [0.75191, 0.75573, 0.75382, 0.73855, 0.74809, 0.75477]
RandomForestOneCNN = [0.74427, 0.73282, 0.73378, 0.75095, 0.74046, 0.7376]

LassoCVElasticNetCVTwoCNN = [0.71947, 0.73855, 0.72615, 0.71565, 0.72805, 0.74809]
RandomForestTwoCNN = [0.71851, 0.70706, 0.71756, 0.73664, 0.71756, 0.73187]

LassoCVElasticNetCVKNN = [0.70515, 0.71851, 0.71565, 0.69943, 0.71279, 0.72137]
RandomForestKNN = [0.70802, 0.70611, 0.69752, 0.70802, 0.7271, 0.72996]

LassoCVElasticNetCVSVM = [0.75859, 0.75954, 0.75763, 0.74332, 0.75477, 0.75859]
RandomForestSVM = [0.74332, 0.73760, 0.7395, 0.74809, 0.74809, 0.75]

plt.rcParams["figure.figsize"] = [8, 6]
fig, ax = plt.subplots(1, 1)

ax.plot(x_10, LassoCVElasticNetCVLSTM, color='#f37649', linewidth=2, linestyle='-', marker='+', ms=6,
        label='LassoCV-ElasticNetCV LSTM')
ax.plot(x_10, RandomForestLSTM, color='#f37649', linewidth=2, linestyle='-', marker='s', ms=5,
        label='RandomForest LSTM')
ax.plot(x_10, LassoCVElasticNetCVOneCNN, color='#5fc6c9', linewidth=2, linestyle='-', marker='+', ms=6,
        label='LassoCV-ElasticNetCV 1D-CNN')
ax.plot(x_10, RandomForestOneCNN, color='#5fc6c9', linewidth=2, linestyle='-', marker='s', ms=5,
        label='RandomForest 1D-CNN')
ax.plot(x_10, LassoCVElasticNetCVTwoCNN, color='#fac00f', linewidth=2, linestyle='-', marker='+', ms=6,
        label='LassoCV-ElasticNetCV 2D-CNN')
ax.plot(x_10, RandomForestTwoCNN, color='#fac00f', linewidth=2, linestyle='-', marker='s', ms=5,
        label='RandomForest 2D-CNN')
ax.plot(x_10, LassoCVElasticNetCVKNN, color='#45596d', linewidth=2, linestyle='-', marker='+', ms=6,
        label='LassoCV-ElasticNetCV KNN')
ax.plot(x_10, RandomForestKNN, color='#45596d', linewidth=2, linestyle='-', marker='s', ms=5, label='RandomForest KNN')
ax.plot(x_10, LassoCVElasticNetCVSVM, color='#015699', linewidth=2, linestyle='-', marker='+', ms=6,
        label='LassoCV-ElasticNetCV SVM')
ax.plot(x_10, RandomForestSVM, color='#015699', linewidth=2, linestyle='-', marker='s', ms=5, label='RandomForest SVM')

font1 = {'size': 8.4}
ax.legend(loc="lower right", prop=font1)
ax.set_xlabel('Number of Features', fontsize=15)
ax.set_ylabel('Classification Accuracy', fontsize=15)
plt.xticks(fontsize=15)  # x轴上的标签旋转45度
plt.yticks(fontsize=15)
plt.ylim(0.60, 0.80)
plt.show()
