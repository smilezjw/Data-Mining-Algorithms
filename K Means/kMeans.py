# coding=utf8

__author__ = 'smilezjw'

from numpy import *
import numpy
import math

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return mat(dataMat)

# 计算欧氏距离
def distEclud(vecA, vecB):
    return math.sqrt(sum(numpy.power(vecA - vecB, 2)))

# 随机生成初始化k个类的质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in xrange(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # 生成的随机质心在每一列的最小值和最大值之间
        # random.rand()生成在0到1之间的正态分布的随机数
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    # clusterAssment记录每个点所属的质心的索引，以及该点和质心之间的距离
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in xrange(m):
            minDist = inf
            minIndex = -1
            for j in xrange(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 修改了这个点所属的质心
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            # nonzero(clusterAssment[:, 0].A == cent) 返回m个点中属于cent这个质心的点的下标
            # dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]] 找到m个点中属于cent这个质心的点
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 按照纵轴计算当前簇的所有点的平均坐标作为新的质心
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    randCentroids = randCent(dataMat, 2)
    print kMeans(dataMat, 4)