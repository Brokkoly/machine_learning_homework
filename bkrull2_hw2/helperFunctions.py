# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:18:43 2017

@author: Brendan
"""

#def condMutInfo(indices,givenIndices,data,attributes):
#    poss = attributes[indices,1]
'''
def likelihood(data,attributes):
    #Find number of occurences of each class
    classTotals = np.zeros(np.size(attributes[-1,1]))
    likelihoods = np.zeros(np.size(data,axis=1)*np.size(attributes[-1,1]))
    likelihoods = np.reshape(likelihoods,(np.size(data,axis=1),np.size(attributes[-1,1])))
    #print likelihoods
    #for i in likelihoods:
        #i = np.zeros(np.size(attributes,axis=0)-1)
    #likelihoods = np.reshape(likelihoods,(np.shape(data)[0],np.shape(data)[1]))   
    for i in data:
        for j in np.arange(np.size(attributes[-1,1])):
            if(i[-1] == attributes[-1,1][j]):#TODO fix attributes to be arrays instead of lists
                classTotals[j] +=1
                break
    for i in np.arange(np.size(data,axis = 0)):
        for j in np.arange(np.size(data,axis = 1)-1):
            for k in np.arange(np.size(attributes[-1,1])):
                if(data[i,j] == attributes[-1,1][k]):#TODO maybe fix attributes to be arrays instead of lists
                    likelihoods[i,k]+=1
    print likelihoods                               
    for i in np.arange(np.size(likelihoods,axis = 0)):
        for j in np.arange(np.size(likelihoods,axis = 1)):
            likelihoods[i,j] = float(likelihoods[i,j])/classTotals[j]
    print likelihood
    '''
    
    #for each possibility in each attribute, 
    
    