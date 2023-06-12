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

import tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from utils import sampling, division_X, division_Y, featureSelectionMethod, featureSelectionMethod_dis
from sklearn.model_selection import KFold
from util_read_feature import get_features82


# 随机种子
seed = 5
# 每位患者的记录数量
dataNumber = 14

train_path = '../../../data/Pre-processed outcome and severity dataset (with Time)/train/train.xlsx'
test_path = '../../../data/Pre-processed outcome and severity dataset (with Time)/test/test.xlsx'
df_train_list = []
df_test_list = []
print('2D-CNN读取数据中')

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)
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
        all_features.insert(1, '严重程度（最终）')

        train_data = df_train[all_features]
        test_data = df_test[all_features]

        train_data_X = train_data[x_features]
        test_data_X = test_data[x_features]

        train_data_Y = train_data[['病人ID', '严重程度（最终）']]
        test_data_Y = test_data[['病人ID', '严重程度（最终）']]
        # listname 只是为了打印列名
        listName = train_data_X

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

        X_train = X_train.reshape(X_train.shape[0], dataNumber, featureNumber, 1)
        X_test = X_test.reshape(X_test.shape[0], dataNumber, featureNumber, 1)

        Y_train = np_utils.to_categorical(Y_train, 2)
        Y_test = np_utils.to_categorical(Y_test, 2)

        def build_model():
            tf.compat.v1.reset_default_graph()
            network = input_data(
                shape=[None, X_train.shape[1], X_train.shape[2], 1])
            network = conv_2d(network, 4, 11, strides=4, activation='relu')
            network = max_pool_2d(network, 3, strides=2)
            network = local_response_normalization(network)
            network = conv_2d(network, 12, 5, activation='relu')
            network = max_pool_2d(network, 3, strides=2)
            network = local_response_normalization(network)
            network = conv_2d(network, 48, 3, activation='relu')
            network = conv_2d(network, 48, 3, activation='relu')
            network = conv_2d(network, 36, 3, activation='relu')
            network = max_pool_2d(network, 3, strides=2)
            network = local_response_normalization(network)
            network = fully_connected(network, 576, activation='tanh')
            network = dropout(network, 0.5)
            network = fully_connected(network, 576, activation='tanh')
            network = dropout(network, 0.5)
            network = fully_connected(network, 2, activation='softmax')
            network = regression(network, optimizer="adam",
                                 loss='categorical_crossentropy',
                                 learning_rate=0.005)
            model = tflearn.DNN(network, tensorboard_verbose=3)
            return model


        AlexNet_mdoel = build_model()

        AlexNet_mdoel.fit(X_train, Y_train, n_epoch=30, batch_size=128,
                          snapshot_epoch=False, snapshot_step=None)
        dir_name = "../pth(final)/" + "2D-CNN/" + method + f"/feature_number{featureNumber}"
        # 如果没有这个文件夹则创建
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # 补充完整路径
        pth_path = dir_name + f"/train.tfl"
        AlexNet_mdoel.save(pth_path)
        part_predict = AlexNet_mdoel.predict(X_test)  # part_predict <class 'numpy.ndarray'>  输出[[A,B],...,[],[],[]]
        part_predict = part_predict.argmax(axis=1)  # 当axis=1,是在行中比较，选出最大的列索引    input:array[[0, 6, 2], [3, 4, 5]]  output:[1 2]
        part_predict = part_predict.tolist()  # Y_predict是list类型,为了统一，part_predict要从ndarray->list
        Y_predict.extend(part_predict)

        # Y[test_index]  <class 'numpy.ndarray'>   [[1. 0.]  ... [1. 0.] [1. 0.] [1. 0.]]
        part_true = Y_test.argmax(axis=1)
        part_true = part_true.tolist()  # Y_true是list类型,为了统一，part_true要从ndarray->list
        Y_true.extend(part_true)

        accuracy = AlexNet_mdoel.evaluate(X_test, Y_test)  # type(accuracy):list
        accuracy = float(accuracy[0])  # list 转float纯数字
        print('Classification accuracy:', accuracy * 100)
        scoreSet.append(accuracy)

        part_predict_pro = AlexNet_mdoel.predict(X_test)
        Y_predict_probability.extend(part_predict_pro)

        from sklearn.metrics import classification_report, confusion_matrix

        if flag:
            sys.stdout = open(f"../result/model result(final)/2D-CNN.txt", "w")
            flag = False
        else:
            sys.stdout = open(f"../result/model result(final)/2D-CNN.txt", "a")

        print("=" * 60 + method + "=" * 60)
        print("="*40 + f"特征选择数量:{featureNumber}" + "="*40)
        print("=" * 30 + "特征" + "=" * 30)
        print(feature)
        print('2D-CNN (disease severity):')
        print('准确率集合', scoreSet)
        print('准确率平均值%.5f' % np.mean(scoreSet))
        print(classification_report(Y_true, Y_predict, digits=5))

        confusion_matrix2 = confusion_matrix(Y_true, Y_predict)
        print('2D-CNN:', confusion_matrix2)

        sys.stdout.close()  # 关闭文件
        sys.stdout = sys.__stdout__

        from keras.utils.np_utils import to_categorical

        Y_true = to_categorical(Y_true)

        Y_true = np.array(Y_true)
        Y_predict = np.array(Y_predict)
        Y_predict_probability = np.array(Y_predict_probability)

        if not os.path.exists(f"../result/numpy data/2D-CNN/{method}"):
            os.makedirs(f"../result/numpy data/2D-CNN/{method}")

        np.save(f"../result/numpy data/2D-CNN/{method}/2D-CNN(Y_true){featureNumber}.npy", Y_true)
        np.save(f"../result/numpy data/2D-CNN/{method}/2D-CNN(Y_predict_probability){featureNumber}.npy", Y_predict_probability)

