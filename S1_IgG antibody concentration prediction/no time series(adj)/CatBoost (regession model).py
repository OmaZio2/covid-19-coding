import sys

sys.path.append("../..")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error \
    , mean_absolute_percentage_error, r2_score
from itertools import chain
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from util_read_feature import get_features82
from utils import featureSelectionMethod3

seed = 5

train_path = "../../../data/Pre-processed antibody dataset/train/train/antibody_train%d.xlsx"
test_path = "../../../data/Pre-processed antibody dataset/train/test/antibody_test%d.xlsx"
df_train_list = []
df_test_list = []

print('CatBoost读取数据中')
for i in range(1, 6):
    df_train = pd.read_excel(train_path % i)
    df_test = pd.read_excel(test_path % i)
    # 将每一折的数据都保存到列表中，方便使用
    df_train_list.append(df_train)
    df_test_list.append(df_test)
print('读取数据完成')

CatBoost_path = '../result/feature selection result/CatBoost_result.txt'
LightGBM_path = '../result/feature selection result/LightGBM_result.txt'
XGBoost_path = '../result/feature selection result/XGBoost_result.txt'
features = get_features82(CatBoost_path, LightGBM_path, XGBoost_path, 30)
print('读取特征完成')

for Method in range(1, 4):
    flag = True
    method = featureSelectionMethod3(Method)
    feature = features[Method - 1][:]
    print("=" * 60 + method + "=" * 60)

    Y_predict = []
    Y_true = []

    for index in range(0, 5):
        print("=" * 30 + f"{index + 1}" + "=" * 30)
        df_train = df_train_list[index]
        df_test = df_test_list[index]

        X_train = df_train[feature]
        Y_train = df_train['S1_IgG']

        X_test = df_test[feature]
        Y_test = df_test['S1_IgG']

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)

        CatBoost_Model = CatBoostRegressor(n_estimators=1000, learning_rate=0.01,
                                           eval_metric="MAE",
                                           objective="MAE",
                                           random_seed=seed,
                                           loss_function='MAE',
                                           l2_leaf_reg=5,
                                           depth=5)
        CatBoost_Model.fit(X_train, Y_train, eval_set=(X_test, Y_test), verbose=True)
        part_predict = CatBoost_Model.predict(X_test)
        Y_predict.append(part_predict)
        Y_true.append(Y_test)

    Y_predict = list(chain(*Y_predict))
    Y_true = list(chain(*Y_true))

    plt.figure(figsize=(50, 4))

    if flag:
        sys.stdout = open(f"../result/model result(adjust)/CatBoost/{method}_CatBoost.txt", "w")
        flag = False
    else:
        sys.stdout = open(f"../result/model result(adjust)/CatBoost/{method}_CatBoost.txt", "a")
    print("="*50 + "特征选择方法:" + method + "="*50)
    print('CatBoost分类器')
    print(feature)

    print('MSE {}'.format((mean_squared_error(Y_true, Y_predict))))
    # RMSE 可以调用 mean_squared_error 方法实现, 设置 squared=False 即可;
    print('RMSE {}'.format((mean_squared_error(Y_true, Y_predict, squared=False))))
    print('MAE {}'.format(mean_absolute_error(Y_true, Y_predict)))
    print('MAPE {}'.format(mean_absolute_percentage_error(Y_true, Y_predict)))
    print('r2 {}'.format(r2_score(Y_true, Y_predict)))

    sys.stdout.close()  # 关闭文件
    sys.stdout = sys.__stdout__

    plt.title('S1_IgG Antibody levels prediction based on CatBoost')
    plt.plot(Y_true, "blue")
    plt.plot(Y_predict, "red")
    plt.legend(['True', 'Predict'], loc='best')
    plt.savefig(f"../result/model result(adjust)/CatBoost/{method}_CatBoost.png", dpi=600)
    # plt.show()
