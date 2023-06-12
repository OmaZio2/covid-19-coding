import sys
sys.path.append('../')
import pandas as pd
from utils import del_rows, del_columns, division_X
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np

"""
生成数据集A
"""
# 设定随机种子为5
seed = 5
df = pd.read_excel(r'd:\临床指标简化整理    实验报告(整理版).xlsx')
# ICU经历,发病天数  和   症状程度,临床结局  逻辑强相关      故去掉
df = df.drop(['发病日期', '入院时间', '是否进入ICU', '发病天数'], axis=1)
# 字符型变量转离散型变量
df['性别'] = df['性别'].astype(str).map({'女': 0, '男': 1})
df['临床结局 '] = df['临床结局 '].astype(str).map({'出院': 0, '死亡': 1})
df['严重程度（最终）'] = df['严重程度（最终）'].astype(str).map({'无症状感染者': 0, '轻型': 0, '重型': 1, '危重型': 1})

# """
# 存在部分临床结局,严重程度缺失的行, 去掉
#   axis	0为行 1为列，default 0，数据删除维度
#   how	{‘any’, ‘all’}, default ‘any’，any：删除带有nan的行；all：删除全为nan的行
#   thresh	int，保留至少 int 个非nan行
#   subset	list，在特定列缺失值处理
#   inplace	bool，是否修改源文件
# """
df = df.dropna(axis=0, how='any', thresh=None, subset=['临床结局 ', '严重程度（最终）'], inplace=False)

print('数据大小:', df.shape)
df = del_rows(df)
print('行处理后,数据大小:', df.shape)
df = del_columns(df)
print('列处理后,数据大小:', df.shape)

# 对缺失值进行补中位数
df = df.fillna(df.median(numeric_only=True))

# 将属于同一个病人id的数据汇总在一个列表中，然后所有病人的列表汇总在一个列表中，形成一个嵌套列表，这样数据就可按id划分
df2 = division_X(df)

# 将数据82分
train_temp, test_temp = train_test_split(df2, test_size=0.2, random_state=seed)
train_temp = sorted(train_temp, key=lambda x: x[0][0], reverse=False)
test_temp = sorted(test_temp, key=lambda x: x[0][0], reverse=False)

# 依次取出嵌套列表中的数据，方便对数据进行处理
train2 = []
for i in train_temp:
    for j in i:
        train2.append(j)

train = pd.DataFrame(train2, columns=df.columns)
# 按病人id升序排列
# train = train.sort_values(by='病人ID', ascending=True)
# 将不用标准化的数值去除，只保留需要标准化的
train_drop = train.drop(['病人ID', '临床结局 ', '性别', '严重程度（最终）', '出院/死亡时间', '检测日期',
                         '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)

sc2 = StandardScaler()
features = sc2.fit_transform(train_drop)
data_transform = pd.DataFrame(features, columns=train_drop.columns)
data_transform.insert(0, '糖尿病(0=无，1=有)', train['糖尿病(0=无，1=有)'].tolist())
data_transform.insert(0, '高血压(0=无，1=有)', train['高血压(0=无，1=有)'].tolist())
data_transform.insert(0, '严重程度（最终）', train['严重程度（最终）'].tolist())
data_transform.insert(0, '性别', train['性别'].tolist())
data_transform.insert(0, '临床结局 ', train['临床结局 '].tolist())
data_transform.insert(0, '出院/死亡时间', train['出院/死亡时间'].tolist())
data_transform.insert(0, '检测日期', train['检测日期'].tolist())
data_transform.insert(0, '病人ID', train['病人ID'].tolist())
data_transform.to_excel(
    "D:\Code\covid-19\COVID-19-Prediction-master\data\Pre-processed outcome and severity dataset (with Time)/train/train.xlsx",
    index=False)

# 同上
test2 = []
for i in test_temp:
    for j in i:
        test2.append(j)

test = pd.DataFrame(test2, columns=df.columns)

# test = test.sort_values(by='病人ID', ascending=True)

test_data = test.drop(['病人ID', '临床结局 ', '性别', '严重程度（最终）', '出院/死亡时间', '检测日期',
                       '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)

features = sc2.transform(test_data)
data_transform = pd.DataFrame(features, columns=test_data.columns)
data_transform.insert(0, '糖尿病(0=无，1=有)', test['糖尿病(0=无，1=有)'].tolist())
data_transform.insert(0, '高血压(0=无，1=有)', test['高血压(0=无，1=有)'].tolist())
data_transform.insert(0, '严重程度（最终）', test['严重程度（最终）'].tolist())
data_transform.insert(0, '性别', test['性别'].tolist())
data_transform.insert(0, '临床结局 ', test['临床结局 '].tolist())
data_transform.insert(0, '出院/死亡时间', test['出院/死亡时间'].tolist())
data_transform.insert(0, '检测日期', test['检测日期'].tolist())
data_transform.insert(0, '病人ID', test['病人ID'].tolist())
data_transform.to_excel(
    "D:\Code\covid-19\COVID-19-Prediction-master\data\Pre-processed outcome and severity dataset (with Time)/test/test.xlsx",
    index=False)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)  # 设置五折交叉验证
iter = 1
train_temp = np.array(train_temp)
for train_index, test_index in kf.split(train_temp):
    # 将对80%训练集分割的训练集和测试集取出
    train_data = train_temp[train_index]
    test_data = train_temp[test_index]

    # 因为每个id的数据都整合到一块形成了嵌套列表，所以需要一个一个取出
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

    # train_data = train_data.sort_values(by='病人ID', ascending=True)
    # test_data = test_data.sort_values(by='病人ID', ascending=True)

    train_data_drop = train_data.drop(['病人ID', '临床结局 ', '性别', '严重程度（最终）', '出院/死亡时间', '检测日期',
                                       '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)
    test_data_drop = test_data.drop(['病人ID', '临床结局 ', '性别', '严重程度（最终）', '出院/死亡时间', '检测日期',
                                     '高血压(0=无，1=有)', '糖尿病(0=无，1=有)'], axis=1)
    sc = StandardScaler()

    train_data_sc = sc.fit_transform(train_data_drop)
    train_data_transform = pd.DataFrame(train_data_sc, columns=train_data_drop.columns)
    train_data_transform.insert(0, '糖尿病(0=无，1=有)', train_data['糖尿病(0=无，1=有)'].tolist())
    train_data_transform.insert(0, '高血压(0=无，1=有)', train_data['高血压(0=无，1=有)'].tolist())
    train_data_transform.insert(0, '严重程度（最终）', train_data['严重程度（最终）'].tolist())
    train_data_transform.insert(0, '性别', train_data['性别'].tolist())
    train_data_transform.insert(0, '临床结局 ', train_data['临床结局 '].tolist())
    train_data_transform.insert(0, '出院/死亡时间', train_data['出院/死亡时间'].tolist())
    train_data_transform.insert(0, '检测日期', train_data['检测日期'].tolist())
    train_data_transform.insert(0, '病人ID', train_data['病人ID'].tolist())
    train_data_transform.to_excel(
        f'D:\Code\covid-19\COVID-19-Prediction-master\data\Pre-processed outcome and severity dataset (with Time)/train/train/oas_train{iter}.xlsx',
        index=False)

    test_data_sc = sc.transform(test_data_drop)
    test_data_transform = pd.DataFrame(test_data_sc, columns=test_data_drop.columns)
    test_data_transform.insert(0, '糖尿病(0=无，1=有)', test_data['糖尿病(0=无，1=有)'].tolist())
    test_data_transform.insert(0, '高血压(0=无，1=有)', test_data['高血压(0=无，1=有)'].tolist())
    test_data_transform.insert(0, '严重程度（最终）', test_data['严重程度（最终）'].tolist())
    test_data_transform.insert(0, '性别', test_data['性别'].tolist())
    test_data_transform.insert(0, '临床结局 ', test_data['临床结局 '].tolist())
    test_data_transform.insert(0, '出院/死亡时间', test_data['出院/死亡时间'].tolist())
    test_data_transform.insert(0, '检测日期', test_data['检测日期'].tolist())
    test_data_transform.insert(0, '病人ID', test_data['病人ID'].tolist())
    test_data_transform.to_excel(
        f'D:\Code\covid-19\COVID-19-Prediction-master\data\Pre-processed outcome and severity dataset (with Time)/train/test/oas_test{iter}.xlsx',
        index=False)
    iter += 1

