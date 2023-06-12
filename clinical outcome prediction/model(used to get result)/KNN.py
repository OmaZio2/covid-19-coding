# -*- coding:utf-8 -*-
import sys
# from xml.sax.handler import all_features
sys.path.append('../..')
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from utils import sampling, division_X, division_Y, featureSelectionMethod, featureSelectionMethod_dis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from util_read_feature import get_features82
import os

# 随机种子
seed = 5
# 每位患者的记录数量
dataNumber = 14
train_path = 'C:\\Users\\Win\\Desktop\\dataset\\data\\Pre-processed outcome and severity dataset (with Time)\\82_train\\82_train.xlsx'
test_path = 'C:\\Users\\Win\\Desktop\\dataset\\data\\Pre-processed outcome and severity dataset (with Time)\\82_test.xlsx'
df_train_list = []
df_test_list = []
print('KNN数据读取数据中')

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)
df_train_list.append(df_train)
df_test_list.append(df_test)
print('读取数据完成')

ElasticNetCV_path = r'C:\Users\Win\Desktop\dataset\data\feature selection result(82)\ElasticNetCV_result.txt'
LassoCV_path = r'C:\Users\Win\Desktop\dataset\data\feature selection result(82)\LassoCV_result.txt'
RandomForest_path = r'C:\Users\Win\Desktop\dataset\data\feature selection result(82)\RandomForest_result.txt'
features = get_features82(ElasticNetCV_path, LassoCV_path, RandomForest_path, 10)
print('读取特征完成')
flag = True

for Method in range(2, 4):
    method = featureSelectionMethod_dis(Method)
    print("="*60 + method + "="*60)
    for featureNumber in range(5, 11):
        # knn的五折分数集合
        knn_scoreSet = []
        # knn预测的类别
        knn_Y_predict = []
        # svm的五折分数集合
        # svm_scoreSet = []
        # svm预测的类别
        # svm_Y_predict = []
        # 实际类别
        Y_true = []
        # knn,svm预测的类别概率   绘制ROC PR曲线会用到
        knn_Y_predict_probability = []
        # svm_Y_predict_probability = []

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

        X_train = np.reshape(X_train, (len(X_train), dataNumber * featureNumber))
        X_test = np.reshape(X_test, (len(X_test), dataNumber * featureNumber))

        knn = KNeighborsClassifier(algorithm='kd_tree', p=1, metric='manhattan')
        # svm_ = svm.SVC(probability=True)

        xTrain, xTest = X_train, X_test
        yTrain, yTest = Y_train, Y_test

        # 训练
        # svm_.fit(xTrain, yTrain)
        knn.fit(xTrain, yTrain)

        # sklearn.predict()  模型预测输入样本所属的类别
        knn_part_predict = knn.predict(xTest)  # <class 'numpy.ndarray'>
        knn_part_predict = knn_part_predict.astype(int).tolist()  # float->int   ndarray->list
        knn_Y_predict.append(knn_part_predict)

        # svm_part_predict = svm_.predict(xTest)  # <class 'numpy.ndarray'>
        # svm_part_predict = svm_part_predict.astype(int).tolist()  # float->int   ndarray->list
        # svm_Y_predict.append(svm_part_predict)

        yTest = yTest.tolist()  # ndarray->list
        Y_true.append(yTest)

        # 测试数据类别概率值
        knn_part_predict_probability = knn.predict_proba(xTest)[:, -1]
        knn_part_predict_probability = knn_part_predict_probability.tolist()  # ndarray->list
        knn_Y_predict_probability.extend(knn_part_predict_probability)

        # svm_part_predict_probability = svm_.predict_proba(xTest)[:, -1]
        # svm_part_predict_probability = svm_part_predict_probability.tolist()  # ndarray->list
        # svm_Y_predict_probability.extend(svm_part_predict_probability)

        knn_Y_predict = list(chain.from_iterable(knn_Y_predict))  # [[1],[2],...,[5]] -> [1,2,4,5]
        # svm_Y_predict = list(chain.from_iterable(svm_Y_predict))  # [[1],[2],...,[5]] -> [1,2,4,5]
        Y_true = list(chain.from_iterable(Y_true))  # [[1],[2],...,[5]] -> [1,2,4,5]

        if flag:
            sys.stdout = open(f"../result/model result(final)/KNN(best is 7).txt", "w")
            flag = False
        else:
            sys.stdout = open(f"../result/model result(final)/KNN(best is 7).txt", "a")

        print("=" * 60 + method + "=" * 60)
        print("=" * 40 + f"特征选择数量:{featureNumber}" + "=" * 40)
        print("=" * 30 + "特征" + "=" * 30)
        print(feature)
        print('disease severity prediction (remove ICU and Day):')
        print("KNN")
        print(classification_report(Y_true, knn_Y_predict, digits=5))
        # print("SVM")
        # print(classification_report(Y_true, svm_Y_predict, digits=5))

        confusion_matrix_knn = confusion_matrix(Y_true, knn_Y_predict)
        print('KNN:', confusion_matrix_knn)
        # confusion_matrix_svm = confusion_matrix(Y_true, svm_Y_predict)
        # print('SVM:', confusion_matrix_svm)

        sys.stdout.close()  # 关闭文件
        sys.stdout = sys.__stdout__

        Y_true = np.array(Y_true)
        knn_Y_predict_probability = np.array(knn_Y_predict_probability)
        # svm_Y_predict_probability = np.array(svm_Y_predict_probability)

        if not os.path.exists(f"../result/numpy data/KNN/{method}"):
            os.makedirs(f"../result/numpy data/KNN/{method}")

        np.save(f"../result/numpy data/KNN/{method}/KNN(Y_true){featureNumber}.npy", Y_true)
        np.save(f"../result/numpy data/KNN/{method}/KNN(Y_predict_probability){featureNumber}.npy", knn_Y_predict_probability)
        # np.save(f"../result/numpy data/KNN-SVM/KNN-SVM(svm_Y_predict_probability){featureNumber}.npy", svm_Y_predict_probability)

