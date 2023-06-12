import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn import preprocessing
from itertools import chain

plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
np.set_printoptions(threshold=np.inf)

seed = 5
importance = 0
for i in range(1, 6):
    path_train = f"../../../data/Pre-processed antibody dataset/train/train/antibody_train{i}.xlsx"
    path_test = f"../../../data/Pre-processed antibody dataset/train/test/antibody_test{i}.xlsx"
    df_train = pd.read_excel(path_train)
    X_train = df_train.drop(['S1_IgG', '病人ID'], axis=1)
    Y_train = df_train['S1_IgG']

    df_test = pd.read_excel(path_test)
    X_test = df_test.drop(['S1_IgG', '病人ID'], axis=1)
    Y_test = df_test['S1_IgG']

    model = XGBRegressor(max_depth=10, n_estimators=10000, objective='reg:squarederror',
                         booster='gbtree', eval_metric='mae', learning_rate=0.01, gamma=0.1,
                         tree_method='exact', min_child_weight=1, subsample=0.71,
                         colsample_bytree=1, reg_alpha=0,
                         reg_lambda=10, random_state=0)
    model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], early_stopping_rounds=500,
              verbose=True)
    importance = importance + model.feature_importances_


MinMaxSc = preprocessing.MinMaxScaler()

# np.true_divide 返回除法的浮点数结果而不作截断
# output_importance是十次特征重要性的平均值
output_importance = pd.DataFrame(np.true_divide(importance, 5))
# 因为三种算法 CatBoost XGBoost LightGBM得出的特征重要性量纲差异大  归一化下
output_importance = MinMaxSc.fit_transform(output_importance)


output_importance = list(chain(*output_importance))

output_importance = pd.Series(output_importance, index=X_train.columns)
output_importance = output_importance.sort_values().tail(30)
print('去除ICU and Day')
print(output_importance)

output_importance2 = output_importance.sort_values(ascending=False)

result = ""
for i, j in zip(output_importance2.axes[0], output_importance2):
    j = round(j, 6)
    result += i + "\t" + str(j) + '\n'
result = result.rstrip('\n')
with open(f'../result/feature selection result/XGBoost_result.txt', 'w') as f:
    f.write(result)

# 作图
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)
output_importance.plot(kind="barh")
plt.title("Selections in the XGBoost Model")
plt.savefig(f"../result/feature selection result/XGBoost_result.png", dpi=600)
# plt.show()
