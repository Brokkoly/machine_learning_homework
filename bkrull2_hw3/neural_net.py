import numpy as np
from numpy import exp, array, size, arange, zeros, matrix, empty, reshape, shape
import matplotlib.pyplot as plt
import copy
import pdb
from sys import argv
import arff
import random as rand

def for_range(input_list, start = 0, mod = 0, axis = 0):
    return arange(start, size(input_list, axis) + mod)





def main():


    """
    train_file = argv[1]
    num_folds = argv[2]
    learning_rate = argv[3]
    num_epochs = argv[4]
    """
    train_file = "sonar.arff"
    num_folds = 10
    learning_rate = .1
    num_epochs = 10

    arff_file = arff.load(open(train_file),'rb')
    data = arff_file['data']
    #pdb.set_trace()
    data = array(data)
    #pdb.set_trace()
    features = data[:,:-1]
    correct = data[:,-1]
    attributes = arff_file['attributes']

    print "Data Loaded"

    #pdb.set_trace()


    print "Classes Renamed to 0 and 1"

    folded_data = folds(data,num_folds)
    #pdb.set_trace()
    net_shape = [size(data,1)-1,size(data,1)-1,1]

    print "Data Folded"


    results = []

    accuracies = zeros(num_folds)
    for i in arange(num_folds):
        netty = Net(net_shape,.1)
        fold_indices = np.delete(arange(num_folds),i)
        current_train = [folded_data[iii] for iii in fold_indices]
        current_test = np.reshape(folded_data[i],(size(folded_data[i])/61,61))#TODO get rid of 61 as it is a magic number

        training_data = np.empty((0, size(current_train[0], axis=1)))
        for k in for_range(current_train, axis=0):
            training_data = np.concatenate((training_data, current_train[k]), axis=0)
        for j in arange(num_epochs):

                #pdb.set_trace()
            for k in training_data:
                #netty.propagate(j[:-1])
                correct = k[-1]
                features = k[:-1]
                sigmoids = netty.propagate(features,correct)
                #results.append((correct,sigmoids[-1]))
                netty.back_propagate(sigmoids,correct,features)
        print "Training Done"
        for k in current_test:
            correct = k[-1]
            features = k[:-1]
            sigmoids = netty.propagate(features, correct)
            results.append((correct,sigmoids[-1]))
        print "Fold %d Done!"%i

    print results

def folds(data,num_folds,replacement = False):

    folds_final = [array([])]*num_folds

    data_a = data[data[:, -1] == 0]
    data_b = data[data[:, -1] == 1]

    data_a_shuffled = rand.sample(data_a,size(data_a,axis = 0))
    data_b_shuffled = rand.sample(data_b, size(data_b, axis=0))
    #for i in arange(num_folds):
    #    folds_final = folds_final.append(array([]))

    data_a_and_b = np.concatenate((data_a_shuffled,data_b_shuffled),axis = 0)

    folds_size = size(folds_final,axis = 0)
    current_fold = 0
    for i in for_range(data_a_and_b):
        folds_final[current_fold] = np.append(folds_final[current_fold],data_a_and_b[i],axis = 0)

        current_fold += 1
        if current_fold == folds_size:
            current_fold = 0


    for i in for_range(folds_final,axis = 0):
        folds_final[i] = np.reshape(folds_final[i],(size(folds_final[i],axis = 0)/size(data_a_and_b,axis = 1),size(data_a_and_b,axis = 1)))
        folds_final[i] = rand.sample(folds_final[i],size(folds_final[i],axis = 0))

    return folds_final




def sigmoid(z):
    return 1.0/(1.0+exp(-1*z))


def h(W, b, x, f = sigmoid):
    if type(W) == list:
        W = array(W)
    z = 0
    for i in for_range(W):
        z += W[i]*x[i]
    z+=b
    return f(z)


def J(W, b, x, y, f = sigmoid):
    h_val = h(W, b, x, f)
    ret_val = .5*(abs(h_val-y)**2)
    return ret_val




class Net:


    def __init__(self,net_shape,weight_std_dev = .001):
        self.weights = empty(size(net_shape,axis = 0)-1,dtype = matrix)
        #Weights[i] represents the weights between layer i and i+1]

        self.W1 = np.random.normal(0.0,weight_std_dev,net_shape[0]+1)
        self.W2 = np.random.normal(0.0,weight_std_dev,(net_shape[1]+)








        self.bias_weights = empty(size(net_shape, axis=0)-1, dtype=matrix)
        self.shape = net_shape
        for i in for_range(net_shape[:-1]):
            rand_weights = np.random.normal(0.0,weight_std_dev,net_shape[i]*net_shape[i+1]) #TODO add these numbers as default arguments
            rand_weights = reshape(rand_weights,(net_shape[i],net_shape[i+1]))
            self.weights[i] = matrix(rand_weights,dtype = float)
            rand_weights = np.random.normal(0.0, weight_std_dev, net_shape[i + 1])
            self.bias_weights[i] = matrix(rand_weights,dtype = float)

    def propagate(self,input_vars,correct):
        #sigmoids = empty((size(self.weights,axis = 0),max(self.shape)),dtype = matrix)
        sigmoids = []
        input_vars = matrix(input_vars)
        H1 = np.dot(input_vars,self.weights[0]) + self.bias_weights[0]
        sigmoids.append(sigmoid(H1))
        #sigs = np.squeeze(np.asarray(sigmoid(H1)))
        #for i in for_range(sigs):
        #    sigmoids[0,i] = sigs[i]
        #sigmoids[0] = matrix(sigmoid(np.squeeze(np.asarray(H1))))
        for i in for_range(self.weights)[1:]:
            #pdb.set_trace()
            sigmoids.append(sigmoid(np.dot(sigmoids[i-1],self.weights[i])+self.bias_weights[i]))
        return sigmoids

    def back_propagate(self,sigmoids,correct,input_vars,learning_rate = .1,b = 1):
        old_weights = copy.deepcopy(self.weights)



        #for layer in arange(size(sigmoids,axis = 0)-1,0,step = -1):
        #TODO make this work for multiple layers
        #Calculation of error of output units
        delta_out = np.dot(np.dot(sigmoids[-1][0],(1-sigmoids[-1][0])),(correct - sigmoids[-1]))
        #Determine updated weights going to output units
        for i in for_range(sigmoids[-2]):#TODO add support for multiple layers

            #for i in arange(size(self.weights[-1,j])):
             #   if(layer == size(sigmoids,axis = 0)-1):
            delta_weights = learning_rate*delta_out*sigmoids[-2][0,i]
            self.weights[-1][i,0] += delta_weights

        #Bias Weight update
        delta_bias = learning_rate*delta_out*b
        self.bias_weights[-1] += delta_bias


        for j in arange(size(sigmoids[-2])):
            delta = sigmoids[-2][0,j]*(1-sigmoids[-2][0,j])*delta_out*old_weights[-1][j,0]

            #Update the weights from the starting layer to the hidden layer
            for i in for_range(input_vars):
                delta_weights = learning_rate*delta*input_vars[i]
                self.weights[0][i,j]
            delta_bias = learning_rate*delta*b
            self.bias_weights[-2] += delta_bias























'''
class Perceptron:

    def __init__(self, num_input_weights, bias = False, h_val = 1):
        self.input_weights = np.empty(num_input_weights)
        self.bias = bias
        self.h_val = -1

        if(self.bias):
            self.h_val = h_val
    def calculate_h(self,x_values,f = sigmoid):

        z = 0.0
        for i in for_range(x_values):
            z+= self.input_weights[i]*x_values[i]
        self.h_val = f(z)
        return self.h_val

    def generate_input_weights(self,num_inputs,epsilon = .01,sigma = .001):
        # type: (object, object, object) -> object
        self.input_weights = abs(np.random.normal(epsilon**2,sigma**2,num_inputs))
        self.input_weights = self.input_weights + epsilon**2

    def reset_h(self,new_val = -1):
        self.h_val = new_val


    def get_h(self,x_values, f = sigmoid):
        if(self.h_val==-1):
            self.calculate_h(x_values,f)
        return self.h_val


    def set_input_weights(self, new_input_weights):
        self.input_weights = new_input_weights


    def set_output_weights(self, new_output_weights):
        self.output_weights = new_output_weights



class Net:


    def __init__(self, layer_info):
        print "hello World"
        self.layers = np.empty(size(layer_info, axis=0), dtype = Layer)
        for i in for_range(self.layers):
            if i == 0:
                num_parent_nodes = 0
                self.layers[i] = Layer(layer_info[i],num_parent_nodes,input = True, input_args=zeros(layer_info[i]))
            else:
                num_parent_nodes = layer_info[i-1]
                self.layers[i] = Layer(layer_info[i],num_parent_nodes)


    def get_result(self,input_params):
        self.nodes[0].set_input_params(input_params)
        for i in for_range(self.nodes[0:]):





        #TODO Propagate through and get output results


class Layer:

    def __init__(self, num_nodes, num_parent_nodes, b = 1,input = False, output = False, input_args = []):
        self.input = input
        self.output = output


        nodes = np.empty(num_nodes + 1,dtype=Perceptron) # Type: ndarray[Perceptron]
        if(self.input):
            for i in for_range(nodes[:-1]):
                nodes[i] = Perceptron(0, bias=True, h_val=input_args[i])
            nodes[-1] = Perceptron(0, bias=True, h_val=b)
        #elif(output):
        else:
            for i in for_range(nodes[:-1]):
                nodes[i] = Perceptron(num_parent_nodes)
                nodes[i].generate_input_weights(num_parent_nodes)
            nodes[-1] = Perceptron(0, bias=True, h_val=b)


    def set_input_args(self,input_args,b = 1):

        if(size(input_args) != (size(self.nodes))-1):
            raise ValueError('# of Nodes != # of Input_Args')
        for i in for_range(self.nodes[:-1]):
            self.nodes[i].reset_h(input_args[i])
        self.nodes[-1].reset_h(b)

    def get_h_vals(self,parent_vals):
        results = np.array([])
        for i in for_range(self.nodes):
            results.append(results,self.nodes[i].get_h(parent_vals))
        return results
'''



if __name__=='__main__':
    main()





