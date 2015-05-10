# coding=utf8

__author__ = 'smilezjw'

from math import log
import operator
import pickle
import TreePlotter

# 计算结果分类的熵
def calShannonEnt(dataSet):
    # 计算数据集中实例的总数
    numEntries = len(dataSet)
    labelCounts = {}
    # 字典的键为分类结果yes/no，值为每一类对应的实例的数量
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # H = - sum(p(xi) * log(p(xi), 2))
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 划分数据集
# 把一个特征label的不同取值的实例集合划分出来
def splitDataSet(dataSet, label, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[label] == value:
            reducedFeatVec = featVec[:label]
            reducedFeatVec.extend(featVec[label+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 计算属性数量
    numFeatures = len(dataSet[0]) - 1
    # 计算分类类别的信息熵
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in xrange(numFeatures):
        # 对于feature[i]得到该属性的取值
        featList = set([feature[i] for feature in dataSet])
        newEntropy = 0.0
        for value in featList:
            # 对于每一种属性的每一组取值，对其划分数据集
            subDataset = splitDataSet(dataSet, i, value)
            # H(feature) = sum(N(fi) / N * H(fi))
            # 对每种属性的每一组取值，计算其分类的熵，然后熵乘以这一组取值的概率
            # 计算这种属性的所有取值的熵的和
            prob = len(subDataset) / float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 如果数据集已经处理了所有属性，但是类标签依然不是唯一的，
# 则采用多数表决的方法决定该叶子结点的分类
def majorityCnt(classList):
    classCount= {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建树
def createTree(dataSet, labels):
    classList = [instance[-1] for instance in dataSet]
    # 如果实例的类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有属性了，但是类别标签不唯一，则多数表决
    # 由于已经遍历完所有属性了，因此data中每一行都只是一个类别标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featValues = set([instance[bestFeat] for instance in dataSet])
    for value in featValues:
        sublabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), sublabels)
    return myTree

# 创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 使用决策树进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 使用pickle模块存储决策树
def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

if __name__ == '__main__':
    # dataSet, labels = createDataSet()
    # featureLabels = labels[:]
    # myTree = createTree(dataSet, labels)
    # print myTree
    # print classify(myTree, featureLabels, [1,1])

    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print lensesTree
    TreePlotter.createPlot(lensesTree)