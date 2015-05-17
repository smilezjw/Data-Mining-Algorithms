# coding=utf8

__author__ = 'smilezjw'

from numpy import *
import numpy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        # curLine = line.strip().split('\t')    # 适用于testSet.txt数据集
        curLine = line.strip().split(',')[:4]   # 适用于iris.data数据集
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

# kMeans算法
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
                # 计算第i个数据点和第j个质心之间的欧氏距离
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

# 二分kMeans算法
def biKMeans(dataSet, k, distMeas=distEclud):
    # 计算实例数量
    m = shape(dataSet)[0]
    # clusterAssment记录每个点所属的质心的索引，以及该点和质心之间的距离
    clusterAssment = mat(zeros((m, 2)))
    # 计算整个数据集的质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    print 'centroid0', centroid0
    centList = [centroid0]
    # 遍历数据集中所有点计算每个点到质心的距离的平方误差
    for i in xrange(m):
        clusterAssment[i, 1] = distMeas(mat(centroid0), dataSet[i, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
        for i in xrange(len(centList)):
            # 将每个簇中的所有点用kMeans进行处理，k=2，生成两个簇和对应的质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算划分为两个簇得到的平方误差，以及剩余数据集的平方误差之和
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print 'sseSplit and notSplit', sseSplit + sseNotSplit
            # 如果该划分得到的SSE最小，则本次划分情况保存下来
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i                   # 被划分的簇的质心
                bestNewCents = centroidMat            # 划分后产生的新的两个质心
                bestClustAss = splitClustAss.copy()   # 用copy的方法保存划分后的簇
                lowestSSE = sseSplit + sseNotSplit
                print 'lowestSSE: ', lowestSSE
        # 用kMeans划分两个新的簇，其编号为0和1，需要将这两个簇编号修改为划分的簇的编号，和新添加的簇的编号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # print 'the bestCentToSplit is:', bestCentToSplit
        # print 'the len of bestClustAss is:', len(bestClustAss)
        # 更新质心列表
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        # 更新聚类的簇
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    print 'centList: ', centList
    print 'clusterAssment: ', clusterAssment
    return mat(centList), clusterAssment

# 球面距离计算公式
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return math.acos(a + b) * 6371.0

# # 簇绘图函数
# def clusterClubs(numClust=5):
#     dataList = []
#     fr = open('places.txt', 'r')
#     for line in fr.readlines():
#         line = line.split('\t')
#         dataList.append([float(line[4]), float(line[3])])
#     datMat = mat(dataList)
#     myCentroids, clustAssing = biKMeans(datMat, numClust, distMeas=distSLC)
#     fig = plt.figure()
#     rect = [0.1, 0.1, 0.8, 0.8]
#     scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
#     axprops = dict(xticks=[], yticks=[])
#     ax0 = fig.add_axes(rect, label='ax0', **axprops)
#     imgP = plt.imread('Portland.png')
#     ax0.imshow(imgP)
#     ax1 = fig.add_axes(rect, label='ax1', frameon=False)
#     for i in xrange(numClust):
#         ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
#         markerStyle = scatterMarkers[i % len(scatterMarkers)]
#         ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
#     ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
#     plt.show()


# 簇绘图函数
def drawCluster(datMat, myCentroids, clustAssing, numClust=3):
    # datMat = loadDataSet('iris.data')
    # myCentroids, clustAssing = biKMeans(datMat, numClust, distMeas=distEclud)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    # ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    ax = fig.add_subplot(111, projection='3d')
    for i in xrange(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 2].flatten().A[0],
                   ptsInCurrCluster[:, 2].flatten().A[0], marker=markerStyle, color='blue', s=90)
    ax.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 2].flatten().A[0], myCentroids[:, 2].flatten().A[0], marker='+', color='black', s=300)
    plt.show()

if __name__ == '__main__':
    # dataMat = loadDataSet('testSet2.txt')
    # randCentroids = randCent(dataMat, 2)
    # print kMeans(dataMat, 4)

    # print biKMeans(dataMat, 3)

    #print clusterClubs(5)

    dataMat = loadDataSet('iris.data')
    # print kMeans(dataMat, 3)
    centList, cluster = biKMeans(dataMat, 3)
    clust1 = clust2 = clust3 = 0
    for i in xrange(3):
        clust1 = max(shape(nonzero(cluster[:50, 0].A == i))[1], clust1)
        clust2 = max(shape(nonzero(cluster[50:100, 0].A == i))[1], clust2)
        clust3 = max(shape(nonzero(cluster[100:150, 0].A == i))[1], clust3)
    print clust1, clust2, clust3
    print 'errorRate: ', 1.0 - float(clust1 + clust2 + clust3) / len(dataMat)
    drawCluster(dataMat, centList, cluster)