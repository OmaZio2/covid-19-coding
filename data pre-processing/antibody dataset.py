import sys

# sys.path.append('D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code')
import pandas as pd
from utils import del_rows, del_columns, division_X
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np

"""
生成数据集B
只考虑S1_IgG
"""

# 设定随机种子为5
seed = 5
df = pd.read_excel(r'C:\Users\Win\Desktop\dataset\临床指标简化整理    实验报告.xlsx')
# 临床结局  严重程度（最终） N_IgG  是否进入ICU  发病天数         和             S1_IgG 已知强相关
# 故去掉
df = df.drop(['发病日期', '入院时间', '出院/死亡时间', '检测日期', '临床结局 ', '严重程度（最终）', 'N_IgG', '是否进入ICU', '发病天数'], axis=1)
# 字符型进行映射数值型
df['性别'] = df['性别'].astype(str).map({'女': 0, '男': 1})

# 存在部分'S1_IgG'缺失的行, 去掉
df = df.dropna(axis=0, how='any', thresh=None, subset=['S1_IgG'], inplace=False)

# 缺失值
print('数据大小:', df.shape)
df = del_rows(df)
print('行处理后,数据大小:', df.shape)
df = del_columns(df)
print('列处理后,数据大小:', df.shape)

# 对缺失值进行补中位数
df = df.fillna(df.median(numeric_only=True))

# df2 = df.drop(['病人ID', '性别', '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)

df2 = division_X(df)

train_temp, test_temp = train_test_split(df2, test_size=0.2, random_state=seed)
train_temp = sorted(train_temp, key=lambda x: x[0][0], reverse=False)
test_temp = sorted(test_temp, key=lambda x: x[0][0], reverse=False)

train2 = []
for i in train_temp:
    for j in i:
        train2.append(j)

train = pd.DataFrame(train2, columns=df.columns)

train_drop = train.drop(['病人ID', '性别', '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)

sc1 = StandardScaler()
features = sc1.fit_transform(train_drop)
data_transform = pd.DataFrame(features, columns=train_drop.columns)
data_transform.insert(0, '糖尿病(0=无，1=有)', train['糖尿病(0=无，1=有)'].tolist())
data_transform.insert(0, '高血压(0=无，1=有)', train['高血压(0=无，1=有)'].tolist())
data_transform.insert(0, '性别', train['性别'].tolist())
data_transform.insert(0, '病人ID', train['病人ID'].tolist())
data_transform.to_excel(
    "C:\\Users\\Win\\Desktop\\dataset\\data\\Pre-processed antibody dataset\82_train\\82_train.xlsx",
    index=False)

test2 = []
for i in test_temp:
    for j in i:
        test2.append(j)

test = pd.DataFrame(test2, columns=df.columns)

test_data = test.drop(['病人ID', '性别', '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)

features = sc1.transform(test_data)
data_transform = pd.DataFrame(features, columns=test_data.columns)
data_transform.insert(0, '糖尿病(0=无，1=有)', test['糖尿病(0=无，1=有)'].tolist())
data_transform.insert(0, '高血压(0=无，1=有)', test['高血压(0=无，1=有)'].tolist())
data_transform.insert(0, '性别', test['性别'].tolist())
data_transform.insert(0, '病人ID', test['病人ID'].tolist())
data_transform.to_excel(
    "C:\\Users\\Win\\Desktop\\dataset\\data\\Pre-processed antibody dataset\\82_test.xlsx",
    index=False)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)  # 设置五折交叉验证
iter = 1
train_temp = np.array(train_temp)
for train_index, test_index in kf.split(train_temp):
    train_data = train_temp[train_index]
    test_data = train_temp[test_index]

    train_data_list = []
    for i in train_data:
        for j in i:
            train_data_list.append(j)

    test_data_list = []
    for i in test_data:
        for j in i:
            test_data_list.append(j)

    train_data = pd.DataFrame(train_data_list, columns=df.columns)
    test_data = pd.DataFrame(test_data_list, columns=df.columns)

    train_data_drop = train_data.drop(['病人ID', '性别', '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)
    test_data_drop = test_data.drop(['病人ID', '性别', '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)
    sc = StandardScaler()

    train_data_sc = sc.fit_transform(train_data_drop)
    train_data_transform = pd.DataFrame(train_data_sc, columns=train_data_drop.columns)
    train_data_transform.insert(0, '糖尿病(0=无，1=有)', train_data['糖尿病(0=无，1=有)'].tolist())
    train_data_transform.insert(0, '高血压(0=无，1=有)', train_data['高血压(0=无，1=有)'].tolist())
    train_data_transform.insert(0, '性别', train_data['性别'].tolist())
    train_data_transform.insert(0, '病人ID', train_data['病人ID'].tolist())
    train_data_transform.to_excel(
        f'C:/Users/Win/Desktop/dataset/data/Pre-processed antibody dataset/82_train/train/antibody_train{iter}.xlsx',
        index=False)

    test_data_sc = sc.transform(test_data_drop)
    test_data_transform = pd.DataFrame(test_data_sc, columns=test_data_drop.columns)
    test_data_transform.insert(0, '糖尿病(0=无，1=有)', test_data['糖尿病(0=无，1=有)'].tolist())
    test_data_transform.insert(0, '高血压(0=无，1=有)', test_data['高血压(0=无，1=有)'].tolist())
    test_data_transform.insert(0, '性别', test_data['性别'].tolist())
    test_data_transform.insert(0, '病人ID', test_data['病人ID'].tolist())
    test_data_transform.to_excel(
        f'C:/Users/Win/Desktop/dataset/data/Pre-processed antibody dataset/82_train/test/antibody_test{iter}.xlsx',
        index=False)
    iter += 1

