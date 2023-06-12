import pandas as pd
from utils import division_X,division_Y
"""
原始数据集的统计
患者数量 n = 1054
1054位患者的样本总数  12621
计算男女具体数量
计算临床结局具体数量
计算症状程度具体数量
计算是否进入ICU具体数量
计算是否高血压具体数量
计算是否糖尿病具体数量
计算年龄的均值和标准差(有三位患者的年龄出现缺失)
"""

#设定随机种子为5
seed = 5
df = pd.read_excel(r'D:\临床指标简化整理    实验报告.xlsx')
#字符型变量转离散型变量
df['性别'] = df['性别'].astype(str).map({'女': 0, '男': 1})
df['临床结局 ']=df['临床结局 '].astype(str).map({'出院':0,'死亡':1})
df['严重程度（最终）']=df['严重程度（最终）'].astype(str).map({'无症状感染者':0,'轻型':1,'重型':2,'危重型':3})
df['是否进入ICU']=df['是否进入ICU'].astype(str).map({'否':0,'是':1})
#计算患者总数
#计算男女具体数量
female = 0
male = 0
Sex_data = df[['病人ID','性别']]
X = division_Y(Sex_data)
for i in range(0,len(X)):
    if X[i][1] == 0:
        female = female + 1
    if X[i][1] == 1:
        male = male + 1
print(len(X),'名患者:')
print('女性:',female)
print("%.1f%%" % (female/(female+male)*100))
print('男性:',male)
print("%.1f%%" % (male/(female+male)*100))


#计算临床结局具体数量
survivor = 0
non_survivor = 0
survivor_data = df[['病人ID','临床结局 ']]
X = division_Y(survivor_data)
for i in range(0,len(X)):
    if X[i][1] == 0:
        survivor = survivor + 1
    if X[i][1] == 1:
        non_survivor = non_survivor + 1
print('出院(存活):',survivor)
print("%.1f%%" % (survivor/(survivor+non_survivor)*100))
print('死亡:',non_survivor)
print("%.1f%%" % (non_survivor/(survivor+non_survivor)*100))

#计算症状程度具体数量
asymptomatic_infection = 0
critical = 0
mild = 0
severe = 0
severity_data = df[['病人ID','严重程度（最终）']]
X = division_Y(severity_data)
for i in range(0,len(X)):
    if X[i][1] == 0:
        asymptomatic_infection = asymptomatic_infection + 1
    if X[i][1] == 1:
        critical = critical + 1
    if X[i][1] == 2:
        mild = mild + 1
    if X[i][1] == 3:
        severe = severe + 1

print('无症状感染者',asymptomatic_infection)
print("%.1f%%" % (asymptomatic_infection/(asymptomatic_infection+critical+mild+severe)*100))
print('轻型',critical)
print("%.1f%%" % (critical/(asymptomatic_infection+critical+mild+severe)*100))
print('重型',mild )
print("%.1f%%" % (mild/(asymptomatic_infection+critical+mild+severe)*100))
print('危重型',severe)
print("%.1f%%" % (severe/(asymptomatic_infection+critical+mild+severe)*100))

#计算是否进入ICU具体数量
non_ICU = 0
ICU  = 0
ICU_data = df[['病人ID','是否进入ICU']]
X = division_Y(ICU_data)
for i in range(0,len(X)):
    if X[i][1] == 0:
        non_ICU = non_ICU +1
    if X[i][1] == 1:
        ICU = ICU + 1
print('没有进入ICU:',non_ICU)
print("%.1f%%" % (non_ICU/(non_ICU+ICU)*100))
print('进入ICU:',ICU)
print("%.1f%%" % (ICU/(non_ICU+ICU)*100))



#计算是否高血压具体数量
non_hypertension = 0
hypertension  = 0
hypertension_data = df[['病人ID','高血压(0=无，1=有)']]
X = division_Y(hypertension_data)
for i in range(0,len(X)):
    if X[i][1] == 0:
        non_hypertension = non_hypertension +1
    if X[i][1] == 1:
        hypertension = hypertension + 1
print('没有高血压:',non_hypertension)
print("%.1f%%" % (non_hypertension/(non_hypertension+hypertension)*100))
print('高血压:',hypertension)
print("%.1f%%" % (hypertension/(non_hypertension+hypertension)*100))


#计算是否糖尿病具体数量
non_diabetes = 0
diabetes  = 0
diabetes_data = df[['病人ID','糖尿病(0=无，1=有)']]
X = division_Y(diabetes_data)
for i in range(0,len(X)):
    if X[i][1] == 0:
        non_diabetes = non_diabetes +1
    if X[i][1] == 1:
        diabetes = diabetes + 1
print('没有糖尿病:',non_diabetes)
print("%.1f%%" % (non_diabetes/(non_diabetes+diabetes)*100))
print('糖尿病:',diabetes)
print("%.1f%%" % (diabetes/(non_diabetes+diabetes)*100))

age_list = []
age_data = df[['病人ID','年龄']].fillna(0)   #有三位患者的年龄出现缺失
X = division_Y(age_data)
print(X)
for i in range(0, len(X)):
    age_list.append(X[i][1])
print(age_list)
import numpy as np
# 求均值
arr_mean = np.mean(age_list)
# 求标准差
arr_std = np.std(age_list, ddof=1)
print(arr_mean)
print(arr_std)