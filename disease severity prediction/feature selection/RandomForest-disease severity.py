import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

importance = 0
for x in range(1, 6):
    seed = 5
    path = f'../../../data/Pre-processed outcome and severity dataset (with Time)/train/train/oas_train{x}.xlsx'
    df = pd.read_excel(path)
    df = df.drop(['检测日期', '出院/死亡时间'], axis=1)
    
    # 预测症状程度,不应该知道临床结局
    df = df.drop(['临床结局 '], axis=1)
    X = df.drop(['严重程度（最终）', '病人ID'], axis=1)
    Y = df['严重程度（最终）']

    model = RandomForestClassifier(n_estimators=5000, random_state=seed, n_jobs=-1)
    model.fit(X, Y.astype('int'))
    importance = importance + model.feature_importances_
    print(f"over{x}")

output_importance = pd.Series(np.true_divide(importance, 5), index=X.columns)
output_importance = output_importance.sort_values().tail(10)
print('去掉ICU和Day的数据集')
print('RandomForest')
print(output_importance)

feature_important2 = output_importance.sort_values(ascending=False)

result = ""
for i, j in zip(feature_important2.axes[0], feature_important2):
    result += i + "\t" + str(j) + '\n'
result = result.rstrip('\n')
with open(f'../result/feature selection result/RandomForest_result.txt', 'w') as f:
    f.write(result)

# 作图
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)
output_importance.plot(kind="barh")
plt.title("Selections in the RandomForest Model")
plt.savefig(f"../result/feature selection result/RandomForest_result.png", dpi=600)
plt.show()
