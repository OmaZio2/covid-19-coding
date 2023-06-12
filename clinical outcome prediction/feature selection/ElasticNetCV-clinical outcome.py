import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
from sklearn.linear_model import LassoCV,ElasticNetCV


seed = 5
df = pd.read_excel(r'C:\Users\Win\Desktop\dataset\data\Pre-processed outcome and severity dataset (with Time)\82_train\82_train.xlsx')
df = df.drop(['检测日期','出院/死亡时间'],axis=1)
#预测临床结局,不应该知道症状程度
df = df.drop(['严重程度（最终）'],axis=1)
X = df.drop(['临床结局 ','病人ID'],axis=1)
x_col = X.columns
Y = df['临床结局 ']

#利用ElasticCV模型选取十个最优特征
alphas = np.logspace(-3,1,100)
model =  ElasticNetCV(alphas = alphas, cv = 5, max_iter = 300000).fit(X,Y) #cv, cross-validation
feature = pd.Series(model.coef_, index=X.columns)
feature = abs(feature)
print("ElasticNetCV picked " + str(sum(feature != 0)) + " variables and eliminated the other " +  str(sum(feature == 0)) + " variables")
feature_important = pd.concat([feature.sort_values().tail(10)])
print('去掉ICU和Day的数据集')
print('ElasticNetCV')
print(feature_important)
result = ""
for i, j in zip(feature_important.axes[0], feature_important):
    result += i + "\t" + str(j) + '\n'
with open(f'../feature selection result(82)/ElasticNetCV_result.txt', 'w') as f:
    f.write(result)
#画图部分
matplotlib.rcParams['figure.figsize'] = (11.0, 5.0)
feature_important.plot(kind="barh")
plt.title("Selection in the ElasticNetCV model")
plt.show()


