#coding:utf-8
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir
#inX:用于分类的数据；dataSet:训练的数据集；labels:训练的数据集每行中每个数据对应的标签；k:从数据集中选择最相似的k个数据
from numpy.ma import zeros, array
import matplotlib.pyplot as plt

def classify0(inX, dataSet, labels, k):
    #
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    #将一个矩阵的每一行向量相加[[1,2],[5,5]]axis=1的结果为：[6,7];axis=0结果:[3,10]
    sqDistances = sum(sqDiffMat,axis=1)
    distances = sqDistances**0.5
    #argsort用于把数组从小到大排序，返回其从小到大排完序后的下标值
    sortedDistIndicies = argsort(distances)
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    print(shape(returnMat)[0],shape(returnMat)[1])
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    # minVals = dataSet.min(0)
    minVals =  amin(dataSet,axis=0)
    maxVals = amax(dataSet,axis=0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))

def useMatplotlib(mat,label):

        # x = linspace(0, 2 * pi, 50)
        # plt.plot(x,sin(x))  # 如果没有第一个参数 x，图形的 x 坐标默认为数组的索引
        # plt.show()  # 显示图形

        figure = plt.figure()
        subplot = figure.add_subplot(111)
        plt.xlabel("每周消费冰淇淋公升数")
        plt.ylabel("玩视频游戏所耗百分比")
        #参数s是散点图的每个点的大小，c是每个散点的颜色
        subplot.scatter(mat[:,1],mat[:,2],s=15.0*array(label),c=15.0*array(label))
        plt.show()

if __name__ == "__main__":
        # data_set,labels = createDataSet()
        # classify_ = classify0([0, 0], data_set, labels, 2)
        # print(classify_)
      # mat,label =file2matrix("datingTestSet2.txt")
      # useMatplotlib(mat,label)
      #   datingClassTest()
      #   handwritingClassTest()
      classCount = {"B":2,"c":7,"G":5}
      print( sorted(classCount.items(), key=operator.itemgetter(1), reverse=True))
      print(type (classCount.items()))


