import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
from sklearn.linear_model import ElasticNetCV

seed = 5
path = '../../../data/Pre-processed outcome and severity dataset (with Time)/train/train.xlsx'
df = pd.read_excel(path)
df = df.drop(['检测日期', '出院/死亡时间'], axis=1)

# 预测症状程度,不应该知道 临床结局
# 在特征选择阶段,我们没有利用患者样本的时序性, 而是将每条样本视为一个独立的个体
df = df.drop(['临床结局 '], axis=1)
X = df.drop(['严重程度（最终）', '病人ID'], axis=1)
Y = df['严重程度（最终）']

alphas = np.logspace(-3, 1, 100)
model = ElasticNetCV(alphas=alphas, cv=5, max_iter=300000).fit(X, Y)  # cv, cross-validation
feature = pd.Series(model.coef_, index=X.columns)
print("ElasticNetCV picked " + str(sum(feature != 0)) + " variables and eliminated the other " + str(
    sum(feature == 0)) + " variables")

# 画出特征变量的重要程度并打印，这里面选出前10个重要的特征,由于相关性有正有负,取abs()
# sort_values()升序排列,
# DataFrame.tail(n)据位置返回对象的最后n行
feature = abs(feature)
print('去掉ICU和Day的数据集')
print('ElasticNetCV')
# print(feature.sort_values())
feature_important = pd.concat([feature.sort_values().tail(10)])
print(feature_important)

feature_important2 = feature_important.sort_values(ascending=False)

result = ""
for i, j in zip(feature_important2.axes[0], feature_important2):
    result += i + "\t" + str(j) + '\n'
result = result.rstrip('\n')
with open(f'../result/feature selection result/ElasticNetCV_result.txt', 'w') as f:
    f.write(result)

# 画图部分
matplotlib.rcParams['figure.figsize'] = (11.0, 5.0)
feature_important.plot(kind="barh")
plt.title("Selections in the ElasticNetCV Model")
plt.savefig(f"../result/feature selection result/ElasticNetCV_result.png", dpi=600)
plt.show()
