import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
from utils import del_rows,del_columns

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




#设定随机种子为5
seed = 5
df = pd.read_excel(r'D:\临床指标简化整理    实验报告.xlsx')
#ICU经历,发病天数  和   症状程度,临床结局  逻辑强相关      故去掉
df = df.drop(['发病日期','入院时间','是否进入ICU','发病天数'], axis=1)
df['性别'] = df['性别'].astype(str).map({'女': 0, '男': 1})
df['临床结局 '] = df['临床结局 '].astype(str).map({'出院': 0, '死亡': 1})
df['严重程度（最终）'] = df['严重程度（最终）'].astype(str).map({'无症状感染者': 0, '轻型': 0, '重型': 1, '危重型': 1})


""" 
存在部分临床结局,严重程度缺失的行, 去掉
  axis	0为行 1为列，default 0，数据删除维度
  how	{‘any’, ‘all’}, default ‘any’，any：删除带有nan的行；all：删除全为nan的行
  thresh	int，保留至少 int 个非nan行
  subset	list，在特定列缺失值处理
  inplace	bool，是否修改源文件
"""
df = df.dropna(axis=0, how='any', thresh=None, subset=['临床结局 ', '严重程度（最终）'], inplace=False)

print('数据大小:', df.shape)
df = del_rows(df)
print('行处理后,数据大小:', df.shape)
df = del_columns(df)
print('列处理后,数据大小:', df.shape)
#统计缺失值是到这步统计的
print(df.keys())
print(missing_values_table(df))