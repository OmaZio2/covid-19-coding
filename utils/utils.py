from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import math

"""函数说明: 
       删除所有样本中缺失值超过50%的行
"""


def del_rows(data):
    t = int(0.5 * data.shape[1])
    data = data.dropna(thresh=t)  # 保留至少有50%非空的行
    return data


"""函数说明: 
       删除所有样本中缺失值超过30%的列
"""


def del_columns(data):
    t = int(0.7 * data.shape[0])
    data = data.dropna(thresh=t, axis=1)  # 保留至少有70%个非空的列
    return data


"""函数说明: 
       (根据ID)将每个患者的样本['ID','特征1','特征2',......,'特征n']划分在一起
   
   参数说明:
       data      所有的样本  
          [['ID1','特征1','特征2',......,'特征n'],
           ['ID1','特征1','特征2',......,'特征n'],
           ['ID2','特征1','特征2',......,'特征n'],
           ['ID2','特征1','特征2',......,'特征n'],
           .......
           ]
       
       list_new  返回划分好的样本集合    三维[患者数量,每个患者的样本数,每份样本的特征数]
           [ [['A0001',1,1,1],
              ['A0001',2,2,2],
              ['A0001',3,3,3]], 
             
             [['A0002',1,1,1],
              ['A0002',2,2,2],
              ['A0002',3,3,3]], 
              ...  
            ]            
       因此list_new的元素是每个患者样本集(list_short)
           list_short的元素是该患者的每条样本(这个样本是有携带ID的)
"""
"""
    函数说明：用于对数据进行预处理，提取提前预测时间
"""
def gapDataprocess(df):
    data = df[['病人ID', '临床结局 ', '血_乳酸脱氢酶', '血_中性粒细胞(#)', '血_尿素', '血_超敏C反应蛋白', '血_血小板计数',
                '血_单核细胞(%)', '血_淋巴细胞(#)', '血_白蛋白', '血_血红蛋白', '血_凝血酶原时间', '提前预测时间']]
    data_X = data[['病人ID', '血_乳酸脱氢酶', '血_中性粒细胞(#)', '血_尿素', '血_超敏C反应蛋白', '血_血小板计数',
                   '血_单核细胞(%)', '血_淋巴细胞(#)', '血_白蛋白', '血_血红蛋白', '血_凝血酶原时间']]
    data_Y = data[['病人ID', '临床结局 ']]
    data_gap = data[['病人ID', '提前预测时间']]
    return data,data_X,data_Y,data_gap

def division_X(df):
    # dataFrame和list之间相互转换
    data = df.values.tolist()

    list_new = []
    list_short = []
    # [0,len(data)-1) == [0,len(data)-2]
    for i in range(0, len(data) - 1):
        # 如果i和i+1的ID相同,那就将该条样本添加到list_short中
        if data[i][0] == data[i + 1][0]:
            list_short.append(data[i])
        # 否则将list_short添加到list_new中,并且重置list_short(便于存下一个ID病人的信息)
        else:
            list_short.append(data[i])
            list_new.append(list_short)
            list_short = []

        # i == len(data)-2是特例
        # 如果按for i in range(0,len(data))
        # i的范围是[0,len(data)) == [0,len(data)-1],进行data[i][0]==data[i+1][0]时会出现index越界
        # 因此到i == len(data) -2 时,
        # if data[len(data)-2]的ID  != data[len(data)-1]的ID, list_short已经完成重置了
        # if data[len(data)-2]的ID  == data[len(data)-1]的ID, list_short加入data[i+1]
        if i == len(data) - 2:
            list_short.append(data[i + 1])
            list_new.append(list_short)

    return list_new


"""函数说明: 
          每个患者有多个样本 (根据ID)提取每个患者的标签
          n对n (但是因为同一个患者的每个样本都具有相同的标签) 转化成   n对1
   参数说明:
          data:   所有的样本
             [['ID1','标签'],
              ['ID1','标签']
              ['ID2','标签']
              ['ID2','标签']
              ........ ]
              
          list_new:  [ID,标签]的集合  
              [['A0001', 0], 
               ['A0002', 0], 
               ['A0003', 0], 
                ...]   
"""


def division_Y(df):
    # dataFrame和list之间相互转换
    data = df.values.tolist()

    list_new = []
    list_new.append(data[0])
    # i的范围[0,len(data)-1)  [0,len(data)-2]
    # 如果i!=i+1   添加data[i+1]进列表
    # 1 1 1 2 2 2 3 3 4 5 6 7 7 7
    # 1 2 3 4 5 6 7
    # 交界两个不一样, 添加下一个
    for i in range(0, len(data) - 1):
        if data[i][0] != data[i + 1][0]:
            list_new.append(data[i + 1])
    return list_new


"""函数说明: 
       每一个病人的样本数量不同,通过采样规范到相同的长度  
       d_i = d[i*n/s] i=1,2,3,.....s
   参数说明:
       inputData:       一个病人的样本集合   
             [['特征1','特征2','特征3','特征4'],   #样本1
              ['特征1','特征2','特征3','特征4'],   #样本2
              ['特征1','特征2','特征3','特征4']]   #样本3
       dataNumber:      需要规范化的样本数量
       featureNumber:   一条样本的特征数量
       output:  经过采样之后的一个病人的样本集合
                以iputData为例   featureNumber=4  dataNUmber=6
                output:      [['特征1','特征2','特征3','特征4'],  #样本1
                              ['特征1','特征2','特征3','特征4'],  #样本1
                              ['特征1','特征2','特征3','特征4'],  #样本2
                              ['特征1','特征2','特征3','特征4'],  #样本2
                              ['特征1','特征2','特征3','特征4'],  #样本3
                              ['特征1','特征2','特征3','特征4']]  #样本3
                通过采样将每个患者的样本数量归一化到相同的长度,但是样本之间时间关系并没有被改变
"""


def sampling(inputData, dataNumber, featureNumber):
    output = []
    """构造一个含dataNumber个 [0,0,0，....,featureNumber]的列表
       if dataNumber = 3 and featureNumber = 4
          [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] 
    """
    for i in range(dataNumber):
        subset = []
        for i in range(featureNumber):
            subset.append(0)
        output.append(subset)

    # math.ceil    向上取整 函数返回大于或等于一个给定数字的最小整数。
    # i的范围 [1,dataNumber+1)  [1,dataNUmber]
    for i in range(1, dataNumber + 1):
        # 通过采样公式     将不等长样本规范化到统一长度dataNumber
        # 为什么要减一     实际数值跟计算机的不一样 1  0      2  1      3 2
        output[i - 1] = inputData[math.ceil(i * len(inputData) / dataNumber) - 1]
    return output


"""函数说明: 打印使用的特征选择方法 特征数量 特征
"""


def featureSelectionMethod(Method, featureNumber, dataFrame):
    if Method == 2:
        print("特征选择方法: LassoCV")
        print("特征选择数量: ", featureNumber)
        print(dataFrame.keys())
    if Method == 1:
        print("特征选择方法: ElasticNetCV")
        print("特征选择数量: ", featureNumber)
        print(dataFrame.keys())
    if Method == 3:
        print("特征选择方法: RandomForest")
        print("特征选择数量: ", featureNumber)
        print(dataFrame.keys())


def featureSelectionMethod2(Method):
    if Method == 2:
        return "LassoCV"
    if Method == 1:
        return "ElasticNetCV"
    if Method == 3:
        return "RandomForest"


def featureSelectionMethod_dis(Method):
    if Method == 2:
        return "ElasticNetCV-LassoCV"
    if Method == 3:
        return "RandomForest"


def featureSelectionMethod3(Method):
    if Method == 1:
        return "CatBoost"
    if Method == 2:
        return "LightGBM"
    if Method == 3:
        return "XGBoost"


"""
timeAttention_Two类说明:
   self.kernel和input_shape大小一致
   output_shape = tf.multiply(self.kernel,input_shape) 逐元素相乘
"""


class timeAttention_Two(Layer):
    def __init__(self, **kwargs):  # 初始化方法

        super(timeAttention_Two, self).__init__(**kwargs)  # 必须要的初始化自定义层

    def build(self, input_shape):
        """[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]
         a = [ 1.4288019e-06 3.8838862e-06 1.0557497e-05 2.8698254e-05 7.8009936e-05
               2.1205300e-04 5.7641981e-04 1.5668715e-03 4.2591984e-03 1.1577702e-02
               3.1471454e-02 8.5548282e-02 2.3254436e-01 6.3212109e-01]
      """
        a = tf.nn.softmax(tf.linspace(1.0, 14.0 - 0.0, 14 - 0))  # 1 * 14
        b = tf.tile(input=[a], multiples=[input_shape[2], 1])  # input_shape[2]*14
        c = tf.transpose(b)  # 14*input_shape[2]
        self.kernel = tf.Variable(c)
        print("build: shape of input: ", input_shape)
        print("build: shape of kernel: ", self.kernel)
        super(timeAttention_Two, self).build(input_shape)

    def call(self, x):
        return tf.multiply(self.kernel, x)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]
