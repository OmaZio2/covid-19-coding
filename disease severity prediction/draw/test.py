import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
# matplotlib.rcParams['figure.figsize'] = (11.0, 5.0)
# x = [0.039272921442942685, 0.04069125465427768, 0.04413253371151454, 0.045410442393773245, 0.04740090018586222,
#      0.049240175133031486, 0.04989277962274426, 0.05687881547998108, 0.060863738293789736, 0.1084720617799501]
# y = ["血_尿酸", "血_单核细胞(%)", "血_淋巴细胞(%)", "血_红细胞计数", "血_RBC分布宽度SD",
#      "年龄", "糖尿病(0=无，1=有)", "血_乳酸脱氢酶", "血_白蛋白", "性别"]
# plt.barh(y, x)
# plt.title("Selection in the LassoCV model")
# plt.savefig(f"D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result(82)\LassoCV_result.png", dpi=600)
# # 显示图
# plt.show()

matplotlib.rcParams['figure.figsize'] = (11.0, 5.0)
x = [0.02172825063154236, 0.022250029545980828, 0.024019216667389123, 0.02866117565782162, 0.031399471482165695,
     0.03151801700679757, 0.03632448854386018, 0.03651601014166147, 0.04039297345317475, 0.04514261335674903]
y = ["血_白细胞计数", "血_RBC分布宽度SD", "血_尿酸", "年龄", "血_白蛋白",
     "血_中性粒细胞(#)", "血_乳酸脱氢酶", "血_中性粒细胞(%)", "血_D-D二聚体定量", "血_淋巴细胞(%)"]
plt.barh(y, x)
plt.title("Selection in the RandomForest model")
plt.savefig(f"D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result(82)\RandomForest_result.png", dpi=600)
# 显示图
plt.show()