import numpy as np
import tan
def likelihood(data, attributes):
    # Find number of occurences of each class
    classNum = np.size(attributes[-1, 1])
    classTotals = np.zeros(classNum)

    # likelihoods = np.zeros(np.size(data,axis=1)*classNum)
    # likelihoods = np.reshape(likelihoods,(np.size(data,axis=1),classNum))
    # print likelihoods
    # for i in likelihoods:
    # i = np.zeros(np.size(attributes,axis=0)-1)
    # likelihoods = np.reshape(likelihoods,(np.shape(data)[0],np.shape(data)[1]))
    numAttributes = 0
    for attribute in attributes[:-1]:
        if np.size(attribute[1]) > numAttributes:
            numAttributes = np.size(attribute[1])
    likelihoods = np.zeros((np.size(attributes, axis=0) - 1, numAttributes, classNum))
    # likelihoods = np.empty(np.size(data,axis = 1)-1,dtype = np.ndarray)
    # for attributeNum in np.arange(np.size(attributes,axis = 0)-1):
    #    likelihoods[attributeNum] = np.zeros(np.size(attributes[attributeNum,1]))
    #    for possibilityNum in np.arange(np.size(likelihoods[attributeNum])):
    #        likelihoods[attributeNum][possibilityNum] = []
    #        for classNum in np.arange(classNum):
    #            likelihoods[attributeNum][possibilityNum].append(0.0)

    # finding total number of occurances of each class
    for each_possibility in np.arange(classNum):
        classTotals[each_possibility] = tan.count([attributes[-1,1][each_possibility]],[-1],data,0)
    # finding the number of times each possibility for each attribute occurs for each class
    for each_instance in np.arange(np.size(data, axis=0)):  # each instance
        for each_attribute in np.arange(np.size(data, axis=1) - 1):  # each attribute in each instance
            for each_possibility in np.arange(np.size(attributes[each_attribute, 1])):
                for each_class in np.arange(classNum):  # each possible class outcome
                    likelihood_count = tan.count([attributes[each_attribute,1][each_possibility],attributes[-1,1][each_class]],[each_attribute,-1],data,1)
                    likelihoods[each_attribute,each_possibility,each_class] = likelihood_count
                    '''
                    instanceClass = data[each_instance, -1]
                    dataval = data[each_instance, each_attribute]
                    # checkClass = attributes[-1,1][k]
                    if ((instanceClass == attributes[-1, 1][each_class]) and (
                        dataval == attributes[each_attribute, 1][each_possibility])):
                        likelihoods[each_attribute, each_possibility, each_class] += 1
                        break
                    '''
    fractional_likelihoods = np.zeros_like(likelihoods)
    for each_instance in np.arange(np.size(fractional_likelihoods, axis=0)):
        for each_attribute in np.arange(np.size(attributes[each_instance, 1])):
            for each_possibility in np.arange(classNum):
                fractional_likelihoods[each_instance, each_attribute, each_possibility] = float(likelihoods[each_instance, each_attribute, each_possibility]) / (classTotals[each_possibility]+(np.size(attributes[each_attribute,1]*classNum)))
    #fractional_likelihoods = fractional_likelihoods

    return [fractional_likelihoods, likelihoods, classTotals]


def getPrediction(testInstance, attributes, likelihoods):
    numClass = np.size(attributes[-1, 1])
    results = np.ones(numClass)
    for testAttribute in np.arange(np.size(testInstance) - 1):
        for possibility in np.arange(np.size(attributes[testAttribute, 1])):
            if (testInstance[testAttribute] == attributes[testAttribute, 1][possibility]):
                for classLikelihood in np.arange(numClass):
                    currentLikelihood = likelihoods[testAttribute, possibility]
                    results[classLikelihood] = results[classLikelihood] * currentLikelihood[classLikelihood]

    return interpretResults(results, attributes, testInstance)


def interpretResults(results, attributes, testInstance):
    numClass = np.size(results)
    maxLike = 0.0
    maxLikeIndex = 0
    totalLike = 0.0
    for i in np.arange(numClass):
        totalLike += results[i]
        if (results[i] > maxLike):
            maxLike = results[i]
            maxLikeIndex = i
    prob = maxLike / totalLike
    retVal = str(attributes[-1, 1][maxLikeIndex]) + " " + str(testInstance[-1]) + " " + "{:.16f}".format(prob)

    return [retVal, testInstance[-1] == attributes[-1, 1][maxLikeIndex]]
def getResults(testData, attributes, likelihoods):
    numCorrect = 0
    results=[]
    for i in testData:
        toPrint,correct=getPrediction(i,attributes, likelihoods)
        results = np.append(results,toPrint)
        if(correct):
            numCorrect+=1
    return [results,numCorrect]
