# coding=utf8

__author__ = 'smilezjw'

from numpy import *

# 创建数据集，包含4个样本和2个类
def createDataSet():
    group = array([[1.0, 0.9],
                   [1.0, 1.0],
                   [0.1, 0.2],
                   [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def kNNClassify(newInput, dataSet, labels, k):
    # numpy.shape 返回（数组的行数， 数组的列数）
    # dataset.shape[0]取到样本数
    numSamples = dataSet.shape[0]

    # 第一步：计算欧氏距离
    # numpy.tile()扩充数组元素，(numSamples, 1)表示扩展的行数和列数
    # 数组之间的运算是元素级的
    diff = tile(newInput, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    # 平时用的sum默认axis=0，是元素级相加；axis=1是将矩阵的每一行向量相加
    squaredDist = sum(squaredDiff, axis=1)
    distance = squaredDist ** 0.5

    # 第二步：对两个样本之间的距离进行排序
    # numpy.argsort()返回数组值从小到大排列的索引值的列表
    sortedDistIndices = argsort(distance)
    classCount = {}

    # 第三步：选择距离最小的前k个样本
    for i in xrange(k):
        voteLabel = labels[sortedDistIndices[i]]
        # 第四步：计算每个样本所属类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 第五步：返回所属类别样本数最多的类别
    maxCount = 0
    for label, count in classCount.items():
        if count > maxCount:
            maxCount = count
            result = label
    return result

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    testData = array([1.2, 1.0])
    k = 3
    result = kNNClassify(testData, dataSet, labels, k)
    print 'Test Data is: ', testData, 'and classified to class: ', result

    testData = array([0.1, 0.3])
    result = kNNClassify(testData, dataSet, labels, k)
    print 'Test Data is: ', testData, 'and classified to class: ', result