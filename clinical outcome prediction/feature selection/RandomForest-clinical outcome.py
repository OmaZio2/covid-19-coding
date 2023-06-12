import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold



seed = 5
importance = np.zeros(57)

for i in range(1,6):
    df = pd.read_excel(f'C:/Users/Win/Desktop/dataset/data/Pre-processed outcome and severity dataset (with Time)/82_train/train/oas_train{i}.xlsx')
    df = df.drop(['检测日期','出院/死亡时间'],axis=1)
    #预测临床结局,不应该知道症状程度
    df = df.drop(['严重程度（最终）'],axis=1)
    X = df.drop(['临床结局 ','病人ID'],axis=1)
    Y = df['临床结局 ']
    model = RandomForestClassifier(n_estimators=10000, random_state=5, n_jobs=-1)
    model.fit(X,Y)
    importance = importance + model.feature_importances_
    print(importance)

#np.true_divide 返回除法的浮点数结果而不作截断
output_importance = pd.Series(np.true_divide(importance,5), index=X.columns)
output_importance = output_importance.sort_values().tail(10)
print('去掉ICU和Day的数据集')
print('RandomForest')
print(output_importance)

# output_importance.to_excel(r'C:\Users\Win\Desktop\学习资料\论文\3.新冠预测\covid-coding\COVID-19-Prediction-master\All figures and screenshots in the paper\new_result\特征选择\clinical outcome prediction\RandomForest.xlsx')
result = ""
for i, j in zip(output_importance.axes[0], output_importance):
    result += i + "\t" + str(j) + '\n'
with open(f'../feature selection result(82)/RandomForest_result.txt', 'w') as f:
    f.write(result)
#作图
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)
output_importance.plot(kind="barh")
plt.title("Selections in the RandomForest Model")
plt.show()