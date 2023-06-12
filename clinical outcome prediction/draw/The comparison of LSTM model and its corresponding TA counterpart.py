
import matplotlib.pyplot as plt
x_10 = [5,6,7,8,9,10]



ElasticNetCV_LassoCVLSTM =  [0.96667,  0.96190,	 0.96190,  0.97143 ,0.98095	,0.96667]
RandomForestLSTM = [0.94762	,0.94286,	0.94762	,    0.95238,    0.96667,  0.97143]


 

ElasticNetCV_LassoCVTALSTM =  [0.99048,	0.97619,	0.98095,	0.98095,	0.98095,	0.98095]
RandomForestTALSTM =   [0.97143,	0.95714, 0.96190,	0.97619,	0.98095,	0.97619]



plt.rcParams["figure.figsize"] = [9, 6]
fig, ax = plt.subplots(1, 1)

##E4392E #3979F2
#----------
ax.plot(x_10,ElasticNetCV_LassoCVLSTM,color='#3979F2',linewidth = 2,linestyle='--',marker='o',ms=6,label='ElasticNetCV&LassoCV LSTM')
ax.plot(x_10,RandomForestLSTM,color='#3979F2',linewidth = 1.5,linestyle='--',marker='+',ms=6,label='RandomForest LSTM')
#----------
ax.plot(x_10,ElasticNetCV_LassoCVTALSTM,color='#E4392E',linewidth = 2,linestyle='-',marker='o',ms=6,label='ElasticNetCV_LassoNetCV TA LSTM')
ax.plot(x_10,RandomForestTALSTM,color='#E4392E',linewidth = 2,linestyle='-',marker='+',ms=6,label='RandomForest TA LSTM')

font1 = {'size': 8.4}
ax.legend(loc="lower right",prop=font1)
ax.set_xlabel('Number of Features',fontsize=15)
ax.set_ylabel('Classification Accuracy',fontsize=15)
plt.xticks(fontsize=15)  # x轴上的标签旋转45度
plt.yticks(fontsize=15)
plt.savefig("../result/数据可视化/The comparison of LSTM model and its corresponding TA counterpart.png")
plt.show()

