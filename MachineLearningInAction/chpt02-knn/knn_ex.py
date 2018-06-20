'''
knn basic test:
输入： 训练数据集
输出： 实例x所属的类y
算法： 
       1. 根据给定的距离度量，在训练集T中找出与x最近的k个点，涵盖这k个点的x的邻域记为N(x);
       2. 在N(x)中根据分类决策规则决定x的类别y
'''
from numpy import *
import operator

def classify(inputX, dataSet, label, k):
    '''
    Input:
        inputX:   vector to compare to existing dataset (1xN);
        dataSet:  known data set (MxN);
        label:    data set label (1xM);
        k:        number of neighbors to use
    '''
    # caculate O-distance
    rows = dataSet.shape[0];
    diffMat = tile(inputX,(rows,1)) - dataSet;
    distance = sqrt(sum(diffMat ** 2,1));
    # select kth points with smallest distance
    index = argsort(distance);
    count = {};
    for i in range(k):
        ithlabel = label[index[i]]
        count[ithlabel] = count.get(ithlabel,0) + 1
    sortedClassCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

