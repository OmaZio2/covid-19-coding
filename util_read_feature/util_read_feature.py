# 废弃
def get_features():
    ElasticNetCV_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result\\train%d\ElasticNetCV_result.txt'
    LassoCV_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result\\train%d\LassoCV_result.txt'
    RandomForest_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result\\train%d\RandomForest_result.txt'
    path = [ElasticNetCV_path, LassoCV_path, RandomForest_path]
    path2 = [ElasticNetCV_path, LassoCV_path, RandomForest_path]
    ElasticNetCV_features = []
    LassoCV_features = []
    RandomForest_features = []
    for i in range(1, 6):
        for j in range(0, 3):
            path2[j] = path[j] % i
            temp = []
            with open(path2[j], 'r', encoding='gbk') as f:
                for line in f:
                    temp.append(line.split()[0])
            if j == 0:
                ElasticNetCV_features.append(temp)
            if j == 1:
                LassoCV_features.append(temp)
            if j == 2:
                RandomForest_features.append(temp)
    result = [ElasticNetCV_features, LassoCV_features, RandomForest_features]
    return result


def get_features82(ElasticNetCV_path, LassoCV_path, RandomForest_path, num):
    # ElasticNetCV_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result(82)\\ElasticNetCV_result.txt'
    # LassoCV_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result(82)\\LassoCV_result.txt'
    # RandomForest_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result(82)\\RandomForest_result.txt'
    # 注意，我这里存放的数值是按降序排列的，数值高的在前
    # 将路径都存放在一个列表中
    path = [ElasticNetCV_path, LassoCV_path, RandomForest_path]
    ElasticNetCV_features = []
    LassoCV_features = []
    RandomForest_features = []
    # 依次从路径列表中取出路径，读取特征
    for i in range(0, 3):
        with open(path[i], 'r', encoding='gbk') as f:
            temp = []
            # 逐行读取，并用空格分割，取出第一个值，即特征
            for line in f:
                temp.append(line.split()[0])
            # 将读取的特征存放到相应算法名字的列表中
            if i == 0:
                ElasticNetCV_features = temp[:num]
            if i == 1:
                LassoCV_features = temp[:num]
            if i == 2:
                RandomForest_features = temp[:num]
    # 将不同算法取出的特征都保存到一个列表中
    result = [ElasticNetCV_features, LassoCV_features, RandomForest_features]
    return result


# 废弃
def v_features():
    ElasticNetCV_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result\\train%d\ElasticNetCV_result.txt'
    LassoCV_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result\\train%d\LassoCV_result.txt'
    RandomForest_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result\\train%d\RandomForest_result.txt'
    path = [ElasticNetCV_path, LassoCV_path, RandomForest_path]
    path2 = [ElasticNetCV_path, LassoCV_path, RandomForest_path]
    for i in range(1, 6):
        for j in range(0, 3):
            path2[j] = path[j] % i
            with open(path2[j], 'r', encoding='gbk') as f:
                content = f.read()
            lines = content.split('\n')[0:10]
            lines.reverse()
            with open(path2[j], 'w') as f:
                f.write('\n'.join(lines))


# 将因为我的数值是升序保存的，为了方便按特征数值高低选取故将数值文件重新读取，按数值降序排列
def v_features82():
    ElasticNetCV_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result(82)\\ElasticNetCV_result.txt'
    LassoCV_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result(82)\\LassoCV_result.txt'
    RandomForest_path = 'D:\Code\covid-19\COVID-19-Prediction-master\COVID-19 Prediction Code\disease severity prediction\\feature selection result(82)\\RandomForest_result.txt'
    path = [ElasticNetCV_path, LassoCV_path, RandomForest_path]
    for i in path:
        with open(i, 'r', encoding='gbk') as f:
            content = f.read()
        lines = content.split('\n')[0:10]
        lines.reverse()
        with open(i, 'w') as f:
            f.write('\n'.join(lines))


if __name__ == '__main__':
    v_features82()
    # get_features82(8)
