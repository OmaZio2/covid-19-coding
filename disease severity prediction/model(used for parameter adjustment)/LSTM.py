import sys
from xml.sax.handler import all_features
sys.path.append('../..')
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.compat.v1.Session(config=config))

import numpy as np
import pandas as pd
from utils import sampling, division_X, division_Y, featureSelectionMethod, featureSelectionMethod_dis
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Flatten, Dropout
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from util_read_feature import get_features82
from keras import optimizers

# 随机种子
seed = 5
# 每位患者的记录数量
dataNumber = 14

train_path = '../../../data/Pre-processed outcome and severity dataset (with Time)/train/train/oas_train%d.xlsx'
test_path = '../../../data/Pre-processed outcome and severity dataset (with Time)/train/test/oas_test%d.xlsx'
df_train_list = []
df_test_list = []
print('LSTM读取数据中')
for i in range(1, 6):
    df_train = pd.read_excel(train_path % i)
    df_test = pd.read_excel(test_path % i)
    df_train_list.append(df_train)
    df_test_list.append(df_test)
print('读取数据完成')

ElasticNetCV_path = '../result/feature selection result/ElasticNetCV_result.txt'
LassoCV_path = '../result/feature selection result/LassoCV_result.txt'
RandomForest_path = '../result/feature selection result/RandomForest_result.txt'
features = get_features82(ElasticNetCV_path, LassoCV_path, RandomForest_path, 10)
print('读取特征完成')
flag = True

for Method in range(2, 4):
    method = featureSelectionMethod_dis(Method)
    print("="*60 + method + "="*60)
    for featureNumber in range(5, 11):
        # 存放五折得分
        scoreSet = []
        # 存放五折(每折的预测标签)
        Y_predict = []
        # 存放五折(每折的实际标签)
        Y_true = []
        # 存放五折(每折预测的概率,便于画AUC,PR图)
        Y_predict_probability = []
        print("="*40 + f"特征选择数量:{featureNumber}" + "="*40)
        feature = features[Method - 1][:featureNumber]
        print("=" * 30 + "特征" + "=" * 30)
        print(feature)
        for index in range(0, 5):
            print("=" * 30 + f"{index+1}" + "=" * 30)
            df_train = df_train_list[index]
            df_test = df_test_list[index]

            all_features = feature[:]
            x_features = feature[:]
            x_features.insert(0, '病人ID')
            all_features.insert(0, '病人ID')
            all_features.insert(1, '严重程度（最终）')

            train_data = df_train[all_features]
            test_data = df_test[all_features]

            train_data_X = train_data[x_features]
            test_data_X = test_data[x_features]

            train_data_Y = train_data[['病人ID', '严重程度（最终）']]
            test_data_Y = test_data[['病人ID', '严重程度（最终）']]
            # listname 只是为了打印列名
            listName = train_data_X

            # featureSelectionMethod(Method, featureNumber, listName)

            X_train = division_X(train_data_X)
            X_test = division_X(test_data_X)
            """ [[['A0001',2,3],['A0001',4,5],['A0001',7,8]],[['A0002',10,11],['A0002',13,14]]]
                [[[2, 3], [4, 5], [7, 8]], [[10, 11], [13, 14]]]
                去除ID
            """
            for i in X_train:
                for j in i:
                    j.pop(0)
            for i in X_test:
                for j in i:
                    j.pop(0)
            # 到这步,X是三维的   [患者数量,每个患者的样本数,每份样本的特征数]

            temp1 = division_Y(train_data_Y)
            temp2 = division_Y(test_data_Y)
            """ [['A0001', 0], ['A0002', 0], ........, ['n', 0]]
                 [0,0,.....0]
                 去除ID
            """
            Y_train = []
            Y_test = []
            for i in range(0, len(temp1)):
                Y_train.append(temp1[i][-1])
            for i in range(0, len(temp2)):
                Y_test.append(temp2[i][-1])

            # 由于每个患者的样本数量不同,对每个患者的样本采样sampling,来限定每个患者的样本数量
            for i in range(0, len(X_train)):
                X_train[i] = sampling(X_train[i], dataNumber, featureNumber)  # sampling(样本,限定每个患者的样本数量,特征数量)

            for i in range(0, len(X_test)):
                X_test[i] = sampling(X_test[i], dataNumber, featureNumber)  # sampling(样本,限定每个患者的样本数量,特征数量)

            X_train = np.array(X_train)
            X_test = np.array(X_test)

            Y_train = np.array(Y_train)
            Y_test = np.array(Y_test)

            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
            class_weights = dict(enumerate(class_weights))
            print('类型权重参数:', class_weights)


            def build_model():
                model = Sequential()
                model.add(LSTM(160, activation='relu', return_sequences=True,
                               input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(Dropout(0.5))
                model.add(BatchNormalization())
                model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dropout(0.5))
                model.add(Dense(1, activation='sigmoid'))
                adam = optimizers.Adam(lr=0.003)
                model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
                return model

            dir_name = "../pth(adjust)/" + "LSTM/" + method + f"/feature_number{featureNumber}"

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            pth_path = dir_name + f"/train{index + 1}.h5"

            reduce_lr = ReduceLROnPlateau(factor=0.9, monitor='loss', patience=2, mode='auto')
            checkpoint = ModelCheckpoint(pth_path, save_best_only=False)

            LSTM_model = build_model()
            LSTM_model.fit(X_train, Y_train, epochs=10, batch_size=128,
                           validation_data=(X_test, Y_test),
                           verbose=2, class_weight=class_weights, callbacks=[checkpoint])
            # no time series models.evaluate  输入数据和标签，输出损失值和选定的指标值
            loss, accuracy = LSTM_model.evaluate(X_test, Y_test, verbose=2)

            # 预测  返回的是类别的索引，即该样本所属的类别标签
            # [[0],[0],[0]......[0]]
            part_predict = LSTM_model.predict_classes(X_test)
            # [0,0,0,....,0]
            part_predict = [i for item in part_predict for i in item]

            Y_predict.extend(part_predict)
            Y_true.extend(Y_test)
            print('loss:', loss, 'accuracy:', accuracy * 100)
            scoreSet.append(accuracy * 100)
            part_predict_probability = LSTM_model.predict(X_test).ravel()
            Y_predict_probability.extend(part_predict_probability)

        if flag:
            sys.stdout = open(f"../result/model result(adjust)/LSTM.txt", "w")
            flag = False
        else:
            sys.stdout = open(f"../result/model result(adjust)/LSTM.txt", "a")

        print("=" * 60 + method + "=" * 60)
        print("=" * 40 + f"特征选择数量:{featureNumber}" + "=" * 40)
        print("=" * 30 + "特征" + "=" * 30)
        print(feature)
        print("LSTM (disease severity):")
        print('准确率集合 ', scoreSet)
        print('准确率平均值%.3f ' % np.mean(scoreSet))
        print(classification_report(Y_true, Y_predict, digits=5))
        confusion_matrix_ = confusion_matrix(Y_true, Y_predict)
        print('LSTM:', confusion_matrix_)

        sys.stdout.close()  # 关闭文件
        sys.stdout = sys.__stdout__

