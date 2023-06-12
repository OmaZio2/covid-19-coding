import sys

from keras.callbacks import ModelCheckpoint

sys.path.append("../..")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
from utils import division_X
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Merge
from keras.layers.core import *
from sklearn.metrics import mean_squared_error, mean_absolute_error \
    , mean_absolute_percentage_error, r2_score
from util_read_feature import get_features82
from utils import featureSelectionMethod3
from keras import optimizers


seed = 5

train_path = "../../../data/Pre-processed antibody dataset/train/train.xlsx"
test_path = "../../../data/Pre-processed antibody dataset/test/test.xlsx"
df_train_list = []
df_test_list = []

print('LSTM读取数据中')
df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)
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
    index = 0

    print("=" * 30 + f"{index + 1}" + "=" * 30)
    df_train = df_train_list[index]
    df_test = df_test_list[index]

    X_feature = feature[:]
    X_feature.insert(0, '病人ID')
    X_feature.append('S1_IgG')

    X_train = df_train[X_feature]
    X_test = df_test[X_feature]

    Y_train = df_train[['病人ID', 'S1_IgG']]
    Y_test = df_test[['病人ID', 'S1_IgG']]

    X_train = division_X(X_train)
    for i in X_train:
        for j in i:
            j.pop(0)
    X_test = division_X(X_test)
    for i in X_test:
        for j in i:
            j.pop(0)

    new_X_train = []
    for i in X_train:
        if len(i) != 1:
            new_X_train.append(i)
    X_train = new_X_train

    new_X_test = []
    for i in X_test:
        if len(i) != 1:
            new_X_test.append(i)
    X_test = new_X_test

    new_Y_train = []
    for i in X_train:
        # new_Y 预测每位患者最后一次的抗体水平
        new_Y_train.append(i[len(i) - 1][30])
        # 取完标签 原数据置0
        i[len(i) - 1][30] = 0
    Y_train = new_Y_train

    new_Y_test = []
    for i in X_test:
        # new_Y 预测每位患者最后一次的抗体水平
        new_Y_test.append(i[len(i) - 1][30])
        # 取完标签 原数据置0
        i[len(i) - 1][30] = 0
    Y_test = new_Y_test

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    X_train = sequence.pad_sequences(X_train, maxlen=20, value=0, padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=20, value=0, padding='post')


    def build_model():
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(100))
        model.add(Dropout(0.5))
        model.add(Dense(3))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        adam = optimizers.Adam(lr=0.005)
        model.compile(loss='mae', optimizer=adam)
        return model


    dir_name = "../pth(final)/" + "LSTM/" + method

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    pth_path = dir_name + f"/train.h5"

    checkpoint = ModelCheckpoint(pth_path, save_best_only=False)

    model = build_model()
    model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=2
              , callbacks=[checkpoint])
    Y_predict.extend(model.predict(X_test))
    Y_true.extend(Y_test)

    plt.figure(figsize=(50, 4))

    if flag:
        sys.stdout = open(f"../result/model result(final)/LSTM/{method}_LSTM.txt", "w")
        flag = False
    else:
        sys.stdout = open(f"../result/model result(final)/LSTM/{method}_LSTM.txt", "a")

    print("=" * 50 + "特征选择方法:" + method + "=" * 50)
    print(feature)

    print('MSE {}'.format((mean_squared_error(Y_true, Y_predict))))
    # RMSE 可以调用 mean_squared_error 方法实现, 设置 squared=False 即可;
    print('RMSE {}'.format((mean_squared_error(Y_true, Y_predict, squared=False))))
    print('MAE {}'.format(mean_absolute_error(Y_true, Y_predict)))
    print('MAPE {}'.format(mean_absolute_percentage_error(Y_true, Y_predict)))
    print('r2 {}'.format(r2_score(Y_true, Y_predict)))

    sys.stdout.close()  # 关闭文件
    sys.stdout = sys.__stdout__

    plt.plot(Y_true, "blue")
    plt.plot(Y_predict, "red")
    plt.legend(['True', 'Predict'], loc='best')
    plt.savefig(f"../result/model result(final)/LSTM/{method}_LSTM.png", dpi=600)
    # plt.show()

