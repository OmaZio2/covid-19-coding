import matplotlib.pyplot as plt
x_10 = [5,6,7,8,9,10]


ElasticNetCV_LassoCVKNN =[0.97143,	0.97619,	0.98095,	0.96667,	0.98095,	0.98095]
RandomForestKNN = [0.97143 ,	0.97143,	0.97619 ,	0.98095,	0.98095	,0.98095 ]

ElasticNetCV_LassoCVTwoDCNN =[0.96667,	0.97143,	0.96667,	0.97143,	0.98095,	0.98095]
RandomForestTwoDCNN = [0.94286,	0.95714,	0.96667,	0.97619,	0.96667	,0.96667]



ElasticNetCV_LassoCVTAOneDCNN =[0.97619,	0.96190,	0.97143,	0.97619,	0.97619,	0.97619]
RandomForestTAOneDCNN = [0.94762,	 0.94762,   0.95238,	0.96190,    0.97143,    0.96667]


ElasticNetCV_LassoCVTALSTM =  [0.99048,	0.97619,	0.98095,	0.98095,	0.98095,	0.98095]
RandomForestTALSTM =  [0.97143,	0.95714, 0.96190,	0.97619,	0.98095,	0.97619]








plt.rcParams["figure.figsize"] = [8, 6]
fig, ax = plt.subplots(1, 1)

ax.plot(x_10,ElasticNetCV_LassoCVKNN,color='#5c7a29',linewidth = 2,linestyle='-',marker='+',ms=6,label='ElasticCV&LassoCV KNN')
ax.plot(x_10,RandomForestKNN,color='#5c7a29',linewidth = 2,linestyle='-',marker='o',ms=5,label='RandomForest KNN')

ax.plot(x_10,ElasticNetCV_LassoCVTALSTM,color='#f37649',linewidth = 2,linestyle='-',marker='+',ms=6,label='ElasticCV&LassoCV TA LSTM')
ax.plot(x_10,RandomForestTALSTM,color='#f37649',linewidth = 2,linestyle='-',marker='o',ms=5,label='RandomForest TA LSTM')
#----------
ax.plot(x_10,ElasticNetCV_LassoCVTAOneDCNN,color='#5fc6c9',linewidth = 2,linestyle='-',marker='+',ms=6,label='ElasticCV&LassoCV TA 1D-CNN')
ax.plot(x_10,RandomForestTAOneDCNN,color='#5fc6c9',linewidth = 2,linestyle='-',marker='o',ms=5,label='RandomForest TA 1D-CNN')
#----------
ax.plot(x_10,ElasticNetCV_LassoCVTwoDCNN,color='#fac00f',linewidth = 2,linestyle='-',marker='+',ms=6,label='ElasticCV$LassoCV 2D-CNN')
ax.plot(x_10,RandomForestTwoDCNN,color='#fac00f',linewidth = 2,linestyle='-',marker='o',ms=5,label='RandomForest 2D-CNN')
#----------

font1 = {'size': 8.4}
ax.legend(loc="lower right",prop=font1)
ax.set_xlabel('Number of Features',fontsize=15)
ax.set_ylabel('Classification Accuracy',fontsize=15)
plt.xticks(fontsize=15)  # x轴上的标签旋转45度
plt.yticks(fontsize=15)
plt.ylim(0.90,1)
plt.savefig("../result/数据可视化/classification_accuracy-feature_selection-clinical_outcome.png")
plt.show()
