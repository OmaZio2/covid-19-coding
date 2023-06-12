import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from utils import division_X

seed = 5
Method = 2
df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed antibody dataset.xlsx')
X = df[['病人ID', '血_淋巴细胞(#)', '血_γ-谷氨酰转肽酶', '血_钙', '血_尿酸',
        '血_红细胞计数', '糖尿病(0=无，1=有)', '血_PLT分布宽度', '血_超敏C反应蛋白', '血_总胆固醇', '血_白/球比值',
        '血_白细胞计数', '血_eGFR(基于CKD-EPI方程)', '血_嗜酸细胞(%)', '血_红细胞压积', '血_总蛋白',
        '血_尿素', '血_肌酐', '血_嗜酸细胞(#)', '血_大血小板比率',
        '血_RBC分布宽度SD', '血_碳酸氢根', '性别', '血_氯', '血_乳酸脱氢酶', '血_RBC分布宽度CV',
        '血_球蛋白', '年龄', '血_血小板压积']]
Y = df['S1_IgG']
# (2530, 30)
print(X.shape)

X = division_X(X)
# 使用时序性数据时 我们将样本数量小于1的患者删除了
count = 0
sampleSum = 0
for i in X:
    if len(i) > 1:
        count = count + 1
        sampleSum = sampleSum + len(i)

# 536
# 2086
print(count)
print(sampleSum)
# 时序性数据由   536例患者  2086份样本组成
