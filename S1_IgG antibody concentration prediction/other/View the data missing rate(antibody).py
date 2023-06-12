import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
from utils import del_rows, del_columns


def missing_values_table(df):
    mis_val = df.isnull().sum()  # 总缺失值
    mis_val_percent = 100 * df.isnull().sum() / len(df)  # 缺失值比例
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)  # 缺失值制成表格
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values',
                                                              1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=True).round(1)
    # 缺失值比例列由大到小排序
    print('Your selected dataframe has {} columns.\nThere are {} columns that have missing values.'.format(df.shape[1],
                                                                                                           mis_val_table_ren_columns.shape[
                                                                                                               0]))
    # 打印缺失值信息
    return mis_val_table_ren_columns


# 设定随机种子为5
seed = 5

df = pd.read_excel(r'D:\临床指标简化整理    实验报告.xlsx')
# 只考虑一种抗体
df = df.drop(['发病日期', '入院时间', '出院/死亡时间', '检测日期', '临床结局 ', '严重程度（最终）', 'N_IgG', '是否进入ICU', '发病天数'], axis=1)
df['性别'] = df['性别'].astype(str).map({'女': 0, '男': 1})
# 存在部分'S1_IgG'缺失的行, 去掉
df = df.dropna(axis=0, how='any', thresh=None, subset=['S1_IgG'], inplace=False)

print('数据大小:', df.shape)
df = del_rows(df)
print('行处理后,数据大小:', df.shape)
df = del_columns(df)
print('列处理后,数据大小:', df.shape)
print(df.keys())
print(missing_values_table(df))
