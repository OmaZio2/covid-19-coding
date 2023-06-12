import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import del_rows,del_columns




seed = 5
df = pd.read_excel(r'D:\临床指标简化整理    实验报告.xlsx')
#只考虑一种抗体
df = df.drop(['发病日期','入院时间','出院/死亡时间','检测日期','临床结局 ','严重程度（最终）','N_IgG','是否进入ICU','发病天数'], axis=1)
df['性别'] = df['性别'].astype(str).map({'女': 0, '男': 1})
# 存在部分'S1_IgG'缺失的行, 去掉
df = df.dropna(axis=0, how='any', thresh=None, subset=['S1_IgG'], inplace=False)


print('数据大小:', df.shape)
df = del_rows(df)
print('行处理后,数据大小:', df.shape)
df = del_columns(df)
print('列处理后,数据大小:', df.shape)
#统计缺失值是到这步统计的
#因为我们对数据集补0的操作会影响到这个直线  所以我就把缺失值去掉了
df = df.dropna(axis=0, how='any', thresh=None, subset=['年龄'], inplace=False)
df = df.dropna(axis=0, how='any', thresh=None, subset=['血_RBC分布宽度CV'], inplace=False)
df = df.dropna(axis=0, how='any', thresh=None, subset=['血_RBC分布宽度SD'], inplace=False)
df = df.dropna(axis=0, how='any', thresh=None, subset=['血_血小板计数'], inplace=False)
df = df.dropna(axis=0, how='any', thresh=None, subset=['血_嗜酸细胞(#)'], inplace=False)
df = df.dropna(axis=0, how='any', thresh=None, subset=['血_eGFR(基于CKD-EPI方程)'], inplace=False)
df = df.dropna(axis=0, how='any', thresh=None, subset=['血_乳酸脱氢酶'], inplace=False)
df = df.dropna(axis=0, how='any', thresh=None, subset=['血_球蛋白'], inplace=False)
df = df.dropna(axis=0, how='any', thresh=None, subset=['血_氯'], inplace=False)
df = df.fillna(0)




# 确定画布
plt.figure(figsize=(8, 6))
tips  = df


#PCT,Age,ICU,Day after onset, GLOB,RDW\_CV, LDH, $Cl^{-}$, and Sex
plt.subplot(331)
sns.regplot(x ='年龄',y="S1_IgG",data=tips,fit_reg=True,scatter_kws={ "alpha":0.5,"s":1})
plt.xlabel("Age") # x轴标题
plt.ylabel("S1_IgG") # y轴标题
plt.xticks([])
plt.yticks([])

plt.subplot(332)
sns.regplot(x='血_RBC分布宽度CV',y='S1_IgG',data=tips,fit_reg=False,scatter_kws={ "alpha":0.5,"s":1})
plt.xlabel("RDW_CV") # x轴标题
plt.ylabel("S1_IgG") # y轴标题
plt.xticks([])
plt.yticks([])


plt.subplot(333)
sns.regplot(x='血_RBC分布宽度SD',y='S1_IgG',data=tips,fit_reg=False,scatter_kws={ "alpha":0.5,"s":1})
#sns.boxplot(x='血_RBC分布宽度SD',y='S1_IgG', data=tips,flierprops={"markersize":0.5},meanline=True,showmeans=True)
plt.xlabel("RDW_SD") # x轴标题
plt.ylabel("S1_IgG") # y轴标题
plt.xticks([])
plt.yticks([])



plt.subplot(334)
sns.regplot(x ='血_血小板计数',y="S1_IgG",data=tips,fit_reg=False,scatter_kws={ "alpha":0.5,"s":1})
plt.xlabel("RBC") # x轴标题
plt.ylabel("S1_IgG") # y轴标题
plt.xticks([])
plt.yticks([])


plt.subplot(335)
sns.regplot(x ='血_嗜酸细胞(#)',y="S1_IgG",data=tips,fit_reg=False,scatter_kws={ "alpha":0.5,"s":1})
plt.xlabel("Eos$\#$") # x轴标题
plt.ylabel("S1_IgG") # y轴标题
plt.xticks([])
plt.yticks([])


plt.subplot(336)
sns.regplot(x ='血_eGFR(基于CKD-EPI方程)',y="S1_IgG",data=tips,fit_reg=True,scatter_kws={ "alpha":0.5,"s":1})
plt.xlabel("eGFR(CKD-EPI)") # x轴标题
plt.ylabel("S1_IgG") # y轴标题
plt.xticks([])
plt.yticks([])


plt.subplot(337)
sns.regplot(x ='血_乳酸脱氢酶',y="S1_IgG",data=tips,fit_reg=False,scatter_kws={ "alpha":0.5,"s":1})
plt.xlabel("LDH") # x轴标题
plt.ylabel("S1_IgG") # y轴标题
plt.xticks([])
plt.yticks([])


plt.subplot(338)
sns.regplot(x ='血_球蛋白',y="S1_IgG",data=tips,fit_reg=True,scatter_kws={ "alpha":0.5,"s":1})
plt.xlabel("GLOB") # x轴标题
plt.ylabel("S1_IgG") # y轴标题
plt.xticks([])
plt.yticks([])

plt.subplot(339)
sns.regplot(x='血_氯',y='S1_IgG',data=tips,fit_reg=False,scatter_kws={ "alpha":0.5,"s":1})
#sns.boxplot(x='血_氯',y='S1_IgG', data=tips,flierprops={"markersize":0.5},meanline=True,showmeans=True)
plt.xlabel("Cl-")
plt.ylabel("S1_IgG")
plt.xticks([])
plt.yticks([])
plt.show()
