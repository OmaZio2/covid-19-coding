import sys
from xml.sax.handler import all_features
# 添加sys.path.append是为了使用utils这个工具文件，可根据自己的路径修改，或自己有方法使用亦可删除
# sys.path.append('D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code')
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.compat.v1.Session(config=config))

import numpy as np
import pandas as pd
from utils import sampling, division_X, division_Y, featureSelectionMethod, featureSelectionMethod2,featureSelectionMethod_dis
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling1D, Conv1D, MaxPooling1D
from util_read_feature import get_features82
from keras import optimizers

# 随机种子
seed = 5
# 每位患者的记录数量
dataNumber = 14

# 读取数据集和训练集

train_path = 'C:\\Users\\Win\\Desktop\\dataset\\data\\Pre-processed outcome and severity dataset (with Time)\\82_train\\82_train.xlsx'
test_path = 'C:\\Users\\Win\\Desktop\\dataset\\data\\Pre-processed outcome and severity dataset (with Time)\\82_test.xlsx'
df_train_list = []
df_test_list = []
print('1D-CNN读取数据中')

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)
df_train_list.append(df_train)
df_test_list.append(df_test)
print('读取数据完成')


# 读取不同特征算法选择出来的特征
ElasticNetCV_path = r'C:\Users\Win\Desktop\dataset\data\feature selection result(82)\ElasticNetCV_result.txt'
LassoCV_path = r'C:\Users\Win\Desktop\dataset\data\feature selection result(82)\LassoCV_result.txt'
RandomForest_path = r'C:\Users\Win\Desktop\dataset\data\feature selection result(82)\RandomForest_result.txt'
features = get_features82(ElasticNetCV_path, LassoCV_path, RandomForest_path, 10)
print('读取特征完成')
flag = True

# 1为ELASTICCV算法选取的特征，2为LASSOCV算法选取的特征，3为RANDOMFOREST算法选取的特征
# 因为我这边1和2选取的特征都一样，故从2开始
for Method in range(2, 4):
    method = featureSelectionMethod_dis(Method)
    print("="*60 + method + "="*60)
    for featureNumber in range(5, 11):
        # 存每折分数
        scoreSet = []
        # 存每折预测标签
        Y_predict = []
        # 存每折的正确标签
        Y_true = []
        # 存预测时候的概率值
        Y_predict_probability = []
        print("="*40 + f"特征选择数量:{featureNumber}" + "="*40)
        feature = features[Method - 1][:featureNumber]
        print("=" * 30 + "特征" + "=" * 30)
        print(feature)
        index = 0
        print("=" * 30 + f"{index+1}" + "=" * 30)

        df_train = df_train_list[index]
        df_test = df_test_list[index]

        all_features = feature[:]
        x_features = feature[:]
        x_features.insert(0, '病人ID')
        all_features.insert(0, '病人ID')
        all_features.insert(1, '临床结局 ')

        train_data = df_train[all_features]
        test_data = df_test[all_features]

        train_data_X = train_data[x_features]
        test_data_X = test_data[x_features]

        train_data_Y = train_data[['病人ID', '临床结局 ']]
        test_data_Y = test_data[['病人ID', '临床结局 ']]
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
        # 将标签取出
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

        # 用于处理样本不均衡
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
        class_weights = dict(enumerate(class_weights))
        print('类型权重参数:', class_weights)

        from keras import regularizers


        def build_model():
            model = Sequential()
            model.add(Conv1D(100, 5, activation='relu', input_shape=(X_train.shape[1],
                                                                     X_train.shape[2])))
            model.add(Conv1D(100, 5, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
            model.add(MaxPooling1D(3))
            model.add(Dropout(0.5))
            model.add(GlobalAveragePooling1D())
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            adam = optimizers.Adam(lr=0.003)
            model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
            return model


        from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

        # 指定用于保存最优模型的路径
        dir_name = "../pth(final)/" + "1D-CNN/" + method + f"/feature_number{featureNumber}"
        # 如果没有这个文件夹则创建
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # 补充完整路径
        pth_path = dir_name + f"/train.h5"
        reduce_lr = ReduceLROnPlateau(factor=0.9, monitor='loss', patience=4, mode='auto')
        # 保存最优的模型参数
        checkpoint = ModelCheckpoint(pth_path, save_best_only=False)
        # 构建模型
        OneCNN_model = build_model()
        OneCNN_model.fit(X_train, Y_train, epochs=30, batch_size=128,
                         class_weight=class_weights
                         , verbose=2, callbacks=[checkpoint])
        loss, accuracy = OneCNN_model.evaluate(X_test, Y_test, verbose=2)
        part_predict = OneCNN_model.predict_classes(X_test)
        part_predict = [i for item in part_predict for i in item]

        Y_predict.extend(part_predict)
        Y_true.extend(Y_test)

        print('loss:', loss, 'accuracy:', accuracy)
        scoreSet.append(accuracy * 100)

        # 预测保留的测试数据以生成概率值
        # https://blog.csdn.net/qq_45098842/article/details/105892647
        part_predict_probability = OneCNN_model.predict(X_test).ravel()
        Y_predict_probability.extend(part_predict_probability)

        Y_true = np.array(Y_true)
        Y_predict = np.array(Y_predict)
        Y_predict_probability = np.array(Y_predict_probability)

        np.save(f"../result/numpy data/1D-CNN/{Method} 1D-CNN(Y_true){featureNumber}.npy", Y_true)
        np.save(f"../result/numpy data/1D-CNN/{Method} 1D-CNN(Y_predict_probability){featureNumber}.npy", Y_predict_probability)

        # 将输出重定向到指定文件，用于保存相关数据
        if flag:
            sys.stdout = open(f"../result/model result(final)/1D-CNN(best is 8).txt", "w")
            flag = False
        else:
            sys.stdout = open(f"../result/model result(final)/1D-CNN(best is 8).txt", "a")
        print("=" * 60 + method + "=" * 60)
        print("=" * 40 + f"特征选择数量:{featureNumber}" + "=" * 40)
        print("=" * 30 + "特征" + "=" * 30)
        print(feature)
        print('1D-CNN (disease severity remove ICU and Day):')
        print('准确率集合', scoreSet)
        print('准确率平均值%.2f' % np.mean(scoreSet))
        print(classification_report(Y_true, Y_predict, digits=5))

        confusion_matrix2 = confusion_matrix(Y_true, Y_predict)
        print('1D-CNN:', confusion_matrix2)

        sys.stdout.close()  # 关闭文件
        sys.stdout = sys.__stdout__