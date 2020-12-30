
import numpy as np

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float) # actual维度为B*1, all维度为B*3
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]  #先按pred降序排列，然后按序号升序排列
    totalLosses = all[:, 0].sum() #正样本量
    giniSum = all[:, 0].cumsum().sum() / totalLosses #正样本占比曲线下的面积

    giniSum -= (len(actual) + 1) / 2.   #减去随机分布的结果
    return giniSum / len(actual)    #按样本量归一化

def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)    #相对理想情况的比例
