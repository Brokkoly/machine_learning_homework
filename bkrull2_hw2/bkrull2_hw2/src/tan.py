# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:38:08 2017

@author: Brendan
"""
import numpy as np
from numpy import size
from numpy import shape
from numpy import arange
from numpy import arange
from numpy import ones
from numpy import zeros
#TODO generalize these functions for any number of arguments
from math import log
import Queue
#import priorityqueue as pq


def memoize(f):
   """ Memoization decorator for functions taking one or more arguments. """
   class memoizeDictionary(dict):
       def __init__(self, f):
           self.f = f
       def __call__(self, *args):
           strargs = str(args)
           if self.has_key(strargs):
               return self[strargs]
           else:
               return self.__doit(args, strargs)

       def __doit(self, key, strkey):
           functionResult = self.f(*key)
           self[strkey] = functionResult
           return functionResult
   return memoizeDictionary(f)


def pxandy(events,indices, data):
    #TODO make this function better
    #TODO allow this function to reference an existing table.
    xArrayIndex = indices[0]
    yArrayIndex = indices[1]
    x = events[0]
    y = events[1]

    num = 1;
    for i in arange(size(data,axis = 0)):
        if((x == data[xArrayIndex][i]) and (y == data[yArrayIndex][i])):
            num+=1;
    return num/(size(data[xArrayIndex])+1)
#def getAllProb(dataArray,attributes):
    #for attributes in arange(size(attributes,axis=0)):
        #for possibilities in np.a
def getCondProbability(events_a,events_b, indices_a,indices_b, data, attributes):#p(a|b)
    probxyz = independentProb(np.append(events_a,events_b), np.append(indices_a,indices_b), data, attributes)
    probz = independentProb(events_b,indices_b, data, attributes)
    return probxyz/probz
def getCondMutInfo(indices,data,attributes):
    #Y = events[-1]
    #Index = indices[-1]
    infoGain = 0
    for posX in attributes[indices[0],1]:
        for posY in attributes[indices[1],1]:
            for posZ in attributes[indices[-1],1]:
                events_a = [posX,posY]
                events_b = [posZ]
                xyz = independentProb(np.append(events_a,events_b),indices,data,attributes)
                xgivenz = getCondProbability([events_a[0]],events_b,[indices[0]],[indices[-1]],data,attributes)
                ygivenz = getCondProbability([events_a[1]],events_b,[indices[1]],[indices[-1]],data,attributes)
                xygivenz = getCondProbability(events_a,events_b,indices[:-1],[indices[-1]],data,attributes)
                infoGain+= xyz*log(xygivenz/(xgivenz*ygivenz),2)
    return infoGain


def getAllCondMutInfo(data,attributes):
    numAttributes = size(attributes,axis=0)-1
    mutInfo = zeros((numAttributes,numAttributes))
    for x in arange(numAttributes):
        for x2 in arange(numAttributes):
            if(x == x2):
                mutInfo[x,x2] = -1
                continue
            infoVal = getCondMutInfo([x,x2,numAttributes], data, attributes)
            mutInfo[x,x2]=infoVal
    #print mutInfo
    return mutInfo



#@memoize
def count(events,indices,data,lc):
    num = lc
    allTrue = True
    for each_instance in data:
        for each_index in arange(size(indices)):
            currentTest = each_instance[indices[each_index]]
            if (currentTest != events[each_index]):
                allTrue = False
                break
        if (allTrue):
            num += 1
        allTrue = True
    return num


def independentProb(events, indices, data,attributes):
    lc = 1
    num = count(events,indices,data,lc)

    numLaplaceDen = 1
    for i in indices:
        #attributeSize = size(attributes[i,1])
        numLaplaceDen*=size(attributes[i,1])
    den = size(data,axis=0)+numLaplaceDen
    return float(num)/den



    return float(num)/(size(data,axis = 0)+1)

'''
def primms(mut_info,data,attributes):
    numAttributes = size(attributes,axis = 0)-1
    pqueue = Queue.PriorityQueue()
    startingPriority = 0
    #priorities = zeros(numAttributes)
    mut_info = mut_info
    #for attribute in arange(numAttributes):
    #    priorities[attribute] = startingPriority
    #
    #    pqueue.put((attribute,priorities[attribute].copy))
    #    if startingPriority == 0:
    #        startingPriority = np.inf
    #    pqueue.put((attribute,startingPriority)
    pqueue = pq.Priority_Queue()
    for attribute in arange(numAttributes):
        pqueue.put([startingPriority,attribute])
        if startingPriority == 0:
            startingPriority = -1*np.inf
    parents = ones(numAttributes)*-1
    while not pqueue.empty():
        currentIndex = pqueue.get()

        for v in arange(numAttributes):
            if(v == currentIndex[1]):
                continue
            if(pqueue.contains(v)):
                mut_test = mut_info[int(currentIndex[1])]
                mut_test2 = mut_test[v]
                current_test = currentIndex[0]
                #if(current_test == np.inf):
                #    current_test = 0
                #    current_index = 0
                if(mut_test2 == -1):
                    continue
                if(mut_test2>current_test):
                    parents[v]=currentIndex[1]
                    pqueue.modify_priority(v,mut_info[int(currentIndex[1]),v])


    weights = zeros(numAttributes)
    for x in arange(numAttributes):
        if(parents[x]!=(-1)):
            weights[x] = mut_info[x, int(parents[x])]
    return parents,weights
'''
def primms2(mut_info):
    pqueue = Queue.PriorityQueue()
    mut_info *= -1
    visited = zeros(size(mut_info,axis = 0))
    visited[0] = 1
    parents = ones(size(mut_info,axis = 0))*-1
    for x in arange(size(mut_info,axis = 0)):
        potentialWeight = mut_info[0, x]
        if(potentialWeight!=-1):
            potentialParent = 0
            vertex = x
            pqueue.put((potentialWeight,(vertex,potentialParent)))
    while(not pqueue.empty()):
        node = pqueue.get()
        if(visited[node[1][0]]):
            continue

        else:
            parents[node[1][0]]=node[1][1]
            visited[node[1][0]]=1
            potentialParent = node[1][0]
            for x in arange(size(mut_info, axis=0)):
                potentialWeight = mut_info[potentialParent, x]
                if (potentialWeight != -1):
                    vertex = x
                    pqueue.put((potentialWeight, (vertex, potentialParent)))
    return parents.astype(int)


def tan_likelihoods(parents,test_data,training_data,attributes):
    #likelihoods = zeros((size(parents),size(parents)))
    #classNum = np.size(attributes[-1,1])
    #for i in arange(parents):
    probabilities = zeros((size(test_data,axis = 0),size(attributes[-1,1])))
    parents = np.array(parents,dtype=int)
    totalProbabilities = ones((size(test_data, axis=0), size(attributes[-1, 1])))
    for test_instance in arange(size(test_data,axis = 0)):
        for i in arange(size(parents)):
            parent_tree = [i]
            current_node = i
            indices = []
            while(parent_tree[-1]!=(-1)):

                parent_tree = np.append(parent_tree,parents[current_node])
                current_node = parents[current_node]
            #for i in parent_tree[:-1]:
            #    indices = np.append[indices,i]
            probability = 0
            parent_tree = parent_tree.astype(int)
            test_instance_array = np.array(test_data[test_instance])
            for j in arange(size(attributes[-1,1])):
                events_a = [test_data[test_instance][i]]
                events_b_temp = test_instance_array[parent_tree[1:-1]]
                events_b_temp = np.append(events_b_temp,attributes[-1,1][j])
                events_b = np.reshape(events_b_temp,size(events_b_temp))
                indices_a = [i]
                indices_b = np.reshape(parent_tree[1:],size(parent_tree[1:]))
                totalProbabilities[test_instance,j] *= getCondProbability(events_a,events_b,indices_a,indices_b,training_data,attributes)
    probabilities = np.empty_like(totalProbabilities)
    for i in arange(size(totalProbabilities,axis = 0)):
        total = 0.0
        for j in arange(size(totalProbabilities,axis = 1)):
            total+=totalProbabilities[i,j]
        for j in arange(size(totalProbabilities,axis = 1)):
            un_normalized_probability = totalProbabilities[i,j]
            normalized_probability = un_normalized_probability/total
            probabilities[i,j] = normalized_probability
    return probabilities

def interpretResults(results, attributes, test_instance):
    num_class = np.size(results)
    max_prob = 0
    max_prob_index = -1
    for i in np.arange(num_class):
        if (results[i] > max_prob):
            max_prob = results[i]
            max_prob_index = i
    retVal = str(attributes[-1, 1][max_prob_index]) + " " + str(test_instance[-1]) + " " + "{:.16f}".format(max_prob)

    return [retVal, test_instance[-1] == attributes[-1, 1][max_prob_index]]
def getResults(results, test_data, attributes):
    numCorrect = 0
    ret_val = []
    for i in arange(size(results,axis = 0)):
        toPrint,correct=interpretResults(results[i],attributes, test_data[i])
        ret_val = np.append(ret_val,toPrint)
        if(correct):
            numCorrect+=1
    return [ret_val,numCorrect]


