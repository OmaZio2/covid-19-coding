import numpy as np
import pandas as pd
import warnings
from scipy import stats,integrate
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')   #忽略了警告错误的输出
import seaborn as sns
sns.set_style("darkgrid")
sns.set(color_codes=True)  #set( )设置主题，调色板更常用

seed = 5
df = pd.read_excel(r'D:\pythonProject\COVID-19-Prediction\data pre-processing\Pre-processed antibody dataset.xlsx')

Y = df['S1_IgG']


sns.kdeplot(Y,shade=True)
plt.show()