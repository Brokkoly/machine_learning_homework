# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:38:08 2017

@author: Brendan
"""

import arff
import numpy as np
import tan
import naive
from sys import argv
#import pdb

def main():
    if(len(argv)!=4):
        print "Invalid Input Arguments L"
        exit()
    arff_train = arff.load(open(argv[1],'rb'))
    train_data = arff_train['data']
    attributes = arff_train['attributes']
    arff_test = arff.load(open(argv[2],'rb'))
    test_data = arff_test['data']
    attributes = np.array(attributes, dtype=object)
    #train_data = np.array(train_data, dtype=object)
    if(argv[3]=='t'):
        mut_cond_info = tan.getAllCondMutInfo(train_data,attributes)
        parents = tan.primms2(mut_cond_info)

        probabilities = tan.tan_likelihoods(parents, test_data, train_data, attributes)
        results, num_correct = tan.getResults(probabilities, test_data, attributes)
        output_file_t("lymph_t_output.txt", results, parents, num_correct, attributes)

    elif(argv[3]=='b'):
        #pdb.set_trace()
        likelihoods, totals, class_totals = naive.likelihood(train_data, attributes)
        results, num_correct = naive.getResults(test_data, attributes, likelihoods)
        #pdb.set_trace()
        output_file_b("lymph_b_output.txt", results, num_correct, attributes)
        # likelihoods,totals,classTotals = naive.likelihood(training_data,attributes)
        # results,numCorrect = naive.getResults(test_data,attributes,likelihoods)
        # output_file_b("lymph_b_output.txt", results, numCorrect, attributes)

    else:
        print "Invalid Input Arguments T"
        exit()


def output_file_b(fileName, results, numCorrect, attributes):
    fileOut = open(fileName,'w')
    for attribute in attributes[:-1,0]:
        fileOut.write(attribute)
        fileOut.write(" class")
        fileOut.write("\n")
    fileOut.write("\n")
    for result in results:
        fileOut.write(result)
        fileOut.write("\n")
    fileOut.write("\n")
    fileOut.write(str(numCorrect))


def output_file_t(fileName, results, parents, numCorrect, attributes):
    fileOut = open(fileName,'w')
    parents = parents.astype(int)
    for attribute in np.arange(np.size(attributes,axis = 0)-1):
        fileOut.write(attributes[attribute,0])
        fileOut.write(" ")
        if(parents[attribute] != -1):
            fileOut.write(" ")
            fileOut.write(attributes[parents[attribute],0])
            fileOut.write(" ")
        fileOut.write("class")
        fileOut.write("\n")
    fileOut.write("\n")
    for result in results:
        fileOut.write(result)
        fileOut.write("\n")
    fileOut.write("\n")
    fileOut.write(str(numCorrect))
'''
def training_data_graph_t(data_range,train_data,test_data,attributes):
    test_size = np.size(test_data,axis = 0)
    nrange = np.arange(1,data_range,5)
    num_correct_all = []
    for i in nrange:
        mut_cond_info = tan.getAllCondMutInfo(randomSample(i,train_data), attributes)
        parents = tan.primms2(mut_cond_info)
        probabilities = tan.tan_likelihoods(parents, test_data, training_data, attributes)
        results, num_correct = tan.getResults(probabilities, test_data, attributes)
        #num_to_push = float(num_correct)/
        num_correct_all = np.append(num_correct_all,float(num_correct)/test_size)
        print "Done with n = %d" % i
    plt.figure(figsize = (9,12))
    plt.plot(nrange,num_correct_all)
    plt.xlabel("Number of Training Instances")
    plt.ylabel("Percentage Correct")
    plt.title("Learning Curve For TAN Algorithm")
    plt.savefig("TAN_Learning_Curve.png")


def training_data_graph_b(data_range,train_data,test_data,attributes):
    test_size = np.size(test_data,axis = 0)
    nrange = np.arange(1,data_range,5)
    num_correct_all = []
    for i in nrange:
        likelihoods, totals, class_totals = bayes.likelihood(randomSample(i,train_data), attributes)
        results, num_correct = bayes.getResults(test_data, attributes, likelihoods)
        num_to_push = float(num_correct)/test_size
        num_correct_all = np.append(num_correct_all,num_to_push)
        print "Done with n = %d" %i
    plt.figure(figsize = (9,12))
    plt.plot(nrange,num_correct_all)
    plt.xlabel("Number of Training Instances")
    plt.ylabel("Percentage Correct")
    plt.title("Learning Curve For Naive Bayes Algorithm")
    plt.savefig("naive_bayes_Learning_Curve.png")

def randomSample(n,data):
    samples = np.empty((n,np.size(data,axis = 1)),dtype=object)
    for i in np.arange(np.size(samples,axis = 0)):
        rando = int(np.floor(random.random()*np.size(data,axis = 0)))
        #print data[rando]
        samples[i] = data[rando]
    return samples
'''
'''
arffFile = arff.load(open("lymph_train.arff",'rb'))
#print arff.dumps(data)
#Import arff file. 
#print data['data']
attributes = arffFile['attributes']
training_data = arffFile['data']

attributes =     np.array(attributes,dtype = object)
training_data = np.array(training_data, dtype = object)
'''
#events = [attributes[0,1][1]]
#indices = [0]
#print tan.independentProb(events,indices,data,attributes)

#indices = np.array([0,1,2])
#events = np.array([1,1,1])
#data_data = np.array([[1,1,0],[1,1,1],[1,2,3],[0,0,1],[3,4,1]])

#data_data_prob = tan.independentProb(events,indices,data_data,attributes)

#data_data_prob = tan.getCondProbability(events,indices,data_data,attributes)
#likelihoods,totals,classTotals = naive.likelihood(training_data,attributes)
#rint likelihoods

#start with a disconnected graph
#probabilities = likelihoods/(sum of yes/no likelihoods)

#arffFile2 = arff.load(open("lymph_test.arff",'rb'))
#test_data = arffFile2['data']

#training_data_graph_b(102,training_data,test_data,attributes)
#training_data_graph_t(102,training_data,test_data,attributes)
#training_data_graph_t(10,training_data,test_data,attributes)
#print "Bayes Graph Done"
#print "TAN Graph Done"
#results,numCorrect = naive.getResults(test_data,attributes,likelihoods)
#output_file_b("lymph_b_output.txt", results, numCorrect, attributes)

'''


mut_cond_info = tan.getAllCondMutInfo(training_data, attributes)
print mut_cond_info
print np.shape(mut_cond_info)
for i in np.arange(np.size(mut_cond_info,axis = 0)):
    print mut_cond_info[i,i]

parents = tan.primms2(mut_cond_info)
print parents
for x in np.arange(np.size(parents)):
    if(parents[x]) == -1:
        print attributes[x,0] + " class"
    else:
        print attributes[x,0] + " " + attributes[int(parents[x]),0] + " class"



probabilities = tan.tan_likelihoods(parents, test_data, training_data, attributes)
#output_file_t("tan_output.txt",results,parents,num_correct,attributes)


results,numCorrect = naive.getResults(test_data,attributes,likelihoods)
output_file("lymph_b_output.txt", results, numCorrect, attributes)

'''
if __name__ == '__main__':
  main()