from sklearn.model_selection import StratifiedKFold
def different_Alg(x_data,y_data,n_spilt=5):
    '''
    :param x_data: 输入数据
    :param y_data: 数据的标签信息
    :param sfolder: 进行K折交叉验证
    :return:
    '''
    sfolder = StratifiedKFold(n_splits=n_spilt, random_state=0, shuffle=True)
    data_train=[]
    data_test=[]
    label_train=[]
    label_test=[]

    for train_data, test_data in sfolder.split(x_data, y_data):
        data_train.append(x_data.iloc[train_data])
        data_test.append(x_data.iloc[test_data])
        label_train.append(y_data.iloc[train_data])
        label_test.append(y_data.iloc[test_data])

    return data_train,label_train,data_test,label_test