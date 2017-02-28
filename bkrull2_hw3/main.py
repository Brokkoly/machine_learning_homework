import numpy as np
from numpy import exp
from numpy import ndarray
from numpy import array
from numpy import size
from numpy import arange


def for_range(input_list, start = 0, mod = 0, axis = 0):
    return arange(start, size(input_list, axis) + mod)


def sigmoid(z):
    return 1/(1+exp(z))




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


class Perceptron:

    def __init__(self,num_input_weights, bias = False, bias_val = 1):
        self.input_weights = np.empty(num_input_weights)
        self.bias = bias
        self.h_val = -1

        if(self.bias):
            self.h_val = bias_val
    def calculate_h(self,x_values,f = sigmoid):

        z = 0.0
        for i in for_range(x_values):
            z+= self.input_weights[i]*x_values[i]
        self.h_val = f(z)
        return self.h_val

    def generate_input_weights(self,num_inputs,epsilon = .01,sigma = .001):
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


    def __init__(self):
        print "hello World"








class Layer:



    def __init__(self, num_nodes, num_parent_nodes, b = 1,input = False, output = False, input_args = []):
        self.input = input
        self.output = output


        nodes = np.empty(num_nodes + 1,dtype=Perceptron)
        if(self.input):
            for i in for_range(nodes[:-1]):
                nodes[i] = Perceptron(0,bias=True,bias_val=input_args[i])
            nodes[-1] = Perceptron(0,bias=True,bias_val=b)
        #elif(output):
        else:
            for i in for_range(nodes[:-1]):
                nodes[i] = Perceptron(num_parent_nodes)
                nodes[-1] = Perceptron(0,bias=True, bias_val=b)



    def set_input_args(self,input_args,b = 1):

        if(size(input_args) != size(self.nodes)):
            raise ValueError('# of Nodes != # of Input_Args')
        for i in for_range(self.nodes[:-1]):
            self.nodes[i].reset_h(input_args[i])
        self.nodes[-1].reset_h(b)








