#!/usr/bin/python
# -*- coding: utf-8 -*- 

from  math import log
import operator

#计算信息熵,输入数据集，返回信息熵
def calculateEntropy(dataSet):
	"""
	求当前数据集Y的信息熵
	永远求的是Y|(条件)的熵
	"""
	
	totalNum=len(dataSet)
	labelCounts={}
	for line in dataSet:
		currentLabel=line[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	
	entropy=0.0
	for key in labelCounts:
		prob=float(labelCounts[key])/totalNum
		entropy-=prob*log(prob,2)

	return entropy

#按某一属性的值划分数据集
#输入为数据集，属性的索引，属性的取值
def splitData(dataSet,index,value):
	"""
	求含有当前特征(index)取值(value)的数据集
	此特征其他取值的不保留，且被处理后的数据里没有这个特征
	即Y|X_index == value 的分布律
	"""
	splitDataSet=[]
	for line in dataSet:
		if line[index]==value:
			data=line[:index]
			data.extend(line[index+1:])
			splitDataSet.append(data)
	return splitDataSet


#选择最好的属性来划分数据
#输入数据集，输出最大信息增益的属性索引
def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1 # 去除输出类
	baseEntropy=calculateEntropy(dataSet)
	maxInformationGain=0.0
	bestFeatureIndex=-1

	for index in range(numFeatures):
		featureList=[]
		for line in dataSet:
			if line[index] not in featureList:
				featureList.append(line[index])
		#  记录当下特征的所有取值
		newEntropy=0.0
		for feature in featureList:
			# 对feature的数量求和: Σ(P(X_index == feature) * H(Y|X_index == feature))
			# P(X_index == feature) : prob
			# H(Y|X_index == feature)): calculateEntropy(splitDataSet)
			splitDataSet=splitData(dataSet,index,feature)
			# len(二维list) = 一维list的数量
			prob=float(len(splitDataSet))/len(dataSet)# x_index == feature的个数/所有 X_index的个数
			newEntropy+=calculateEntropy(splitDataSet)*prob
		
		informationGain=baseEntropy-newEntropy
		if informationGain > maxInformationGain:
			maxInformationGain=informationGain
			bestFeatureIndex=index
	return bestFeatureIndex
			
#投票表决函数
#输入为分类列表，输出为多数表决的结果
def voteResult(classList):
	classCounts={}
	for value in classList:
		if value not in classCounts:
			classCounts[value]=0
		classCount[value]+=1
	# 获得了一个dictionary
	
	# operator模块提供的itemgetter函数用于获取对象的哪些维的数据
	# 参数为一些序号（即需要获取的数据在对象中的序号）
	sortClassCounts=sorted(classCounts.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortClassCounts[0][0]

#用递归的办法生成决策树，结果用一个字典来表示
#输入为数据集和每个属性的取值，输出为决策树（用字典表示）
def createDecisionTree(dataSet,labels):
	classList=[line[-1] for line in dataSet] # 获得Y的list
	if len(set(classList))==1:             # 如果分类集合只有一个元素则终止递归
		return classList[0] 

	if len(dataSet[0])==1:                 # 如果所有的属性都使用过了，则终止递归 
		return voteResult(classList)

	bestFeatureIndex=chooseBestFeatureToSplit(dataSet)
	bestFeatureLabel=labels[bestFeatureIndex]
	decisionTree={bestFeatureLabel:{}}
	del(labels[bestFeatureIndex])

	featureList=[line[bestFeatureIndex] for line in dataSet]
	uniqueFeatureList=set(featureList)

	for value in uniqueFeatureList:
		splitDataSet=splitData(dataSet,bestFeatureIndex,value)
		decisionTree[bestFeatureLabel][value]=createDecisionTree(splitDataSet,labels)
		
	return decisionTree




#---------------------------测试--------------------------
#测试数据
def testData():
	dataSet=[[1,1,"yes"],
		 [1,1,"yes"],
		 [1,0,"no"],
	         [0,0,"no"],
	         [0,1,"no"]]
	labels=["no surfacing","flippers"]
	return dataSet, labels

#测试calculateEntropy函数
def test_calculateEntropy():
	dataSet, labels=testData()
	entropy=calculateEntropy(dataSet)
	print (entropy)

#测试splitData函数
def test_splitData():
	dataSet, labels=testData()
	splitDataSet=splitData(dataSet,0,0)
	print (splitDataSet)

#测试chooseBestFeatureToSplit函数
def test_chooseBestFeatureToSplit():
	dataSet, labels=testData()
	bestFeatureIndex=chooseBestFeatureToSplit(dataSet)
	print (bestFeatureIndex)
#测试createDecisionTree函数
def test_createDecisionTree():
	dataSet, labels=testData()
	decisionTree=createDecisionTree(dataSet,labels)
	print (decisionTree)

test_createDecisionTree()
#test_splitData()
#test_chooseBestFeatureToSplit()

