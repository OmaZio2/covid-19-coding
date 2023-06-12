import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

x_10 = [5, 6, 7, 8, 9, 10]

# 已修
ElasticNet_LassoCVOneCNN =[0.95714,0.95714,0.95714,0.96190,0.96190,0.96190]
RandomForestOneCNN =[0.93810,0.94286,0.95714,0.95238,0.95714,0.95714]


ElasticNetCV_LassoCVTAOneCNN =[0.97619,	0.96190,	0.97143,	0.97619,	0.97619,	0.97619]
RandomForestTAOneCNN = [0.94762,	 0.94762,   0.95238,	0.96190,    0.97143,    0.96667]


plt.rcParams["figure.figsize"] = [9, 6]
fig, ax = plt.subplots(1, 1)

##E4392E #3979F2
# ----------
ax.plot(x_10, ElasticNet_LassoCVOneCNN, color='#3979F2', linewidth=2, linestyle='--', marker='+', ms=7, label='ElasticNet&LassoCV 1D-CNN')
ax.plot(x_10, RandomForestOneCNN, color='#3979F2', linewidth=2, linestyle='--', marker='s', ms=6, label='RandomForest 1D-CNN')
# ----------
ax.plot(x_10, ElasticNetCV_LassoCVTAOneCNN, color='#E4392E', linewidth=2, linestyle='-', marker='+', ms=7, label='ElasticNet&LassoCV TA1D-CNN')
ax.plot(x_10, RandomForestTAOneCNN, color='#E4392E', linewidth=2, linestyle='-', marker='s', ms=6, label='RandomForest TA1D-CNN')

font1 = {'size': 8.4}
ax.legend(loc="lower right",prop=font1)
ax.set_xlabel('Number of Features',fontsize=15)
ax.set_ylabel('Classification Accuracy',fontsize=15)
plt.xticks(fontsize=15)  # x轴上的标签旋转45度
plt.yticks(fontsize=15)
plt.savefig("../result/数据可视化/The comparison of 1D-CNN model and its corresponding TA counterpart.png")
plt.show()