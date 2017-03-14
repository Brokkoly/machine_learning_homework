import numpy as np
from numpy import exp, array, size, arange, zeros, matrix, empty, reshape, shape
import matplotlib.pyplot as plt
import copy
import pdb
from sys import argv
import machine_learning_homework.bkrull2_hw3.src.arff
import random as rand
import matplotlib.pyplot as plt


def for_range(input_list, start = 0, mod = 0, axis = 0):
    return arange(start, size(input_list, axis) + mod)





def main():



    train_file = argv[1]
    num_folds = argv[2]
    learning_rate = argv[3]
    num_epochs = argv[4]
    """
    train_file = "sonar.arff"
    num_folds = 10
    learning_rate = .1
    num_epochs = 25
    """
    arff_file = machine_learning_homework.bkrull2_hw3.src.arff.load(open(train_file), 'rb')
    data = arff_file['data']
    #pdb.set_trace()
    data = array(data)
    #pdb.set_trace()
    features = data[:,:-1]
    correct = data[:,-1]
    attributes = arff_file['attributes']



    print_val = get_accuracies(data,attributes,num_epochs,num_folds,learning_rate)



    '''
    epoch_accuracy = np.empty((4,10))
    epochs = [25,50,75,100]
    for i in for_range(epochs):
        epoch_accuracy[i] = get_accuracies(data,attributes,epochs[i],num_folds,learning_rate)

    print_epoch_accuracy(epoch_accuracy,epochs)

    fold_nums = [5, 10, 15, 20, 25]
    fold_accuracy = []
    for i in for_range(fold_nums):
        fold_accuracy.append(get_accuracies(data,attributes,50,fold_nums[i],learning_rate))
    print_fold_accuracy(fold_accuracy,fold_nums)
    '''

'''
    acc,ROC_results = get_accuracies(data,attributes,50,10,learning_rate,ROC = True)
    for result in for_range(ROC_results):
        if (ROC_results[result][2] >= .5):
            ROC_results[result][2] = 1
        else:
            ROC_results[result][2] = 0
    sorted_results = ROC_results[ROC_results[:,3].argsort()]
    get_ROC_results(sorted_results)
'''

def get_ROC_results(ROC_results):
    flips = []
    for a in -for_range(ROC_results):
        if(int(ROC_results[a,1])!=int(ROC_results[a,2])):
            flips.append(-a)

    print flips
    xflips = zeros(size(flips))
    xflips[1:] = flips[:-1]
    print xflips


    plt.figure(figsize = (9,12))
    plt.plot(xflips,flips)
    plt.title('ROC Curve')
    plt.xlabel('# of False Positives')
    plt.ylabel('# of True Positives')

    plt.savefig('ROC.png')


def print_epoch_accuracy(accuracies,epochs):
    plt.figure(figsize = (9,12))
    accuracy_averages = []
    for i in for_range(accuracies):
        accuracy_averages.append(np.average(accuracies[i]))
    plt.plot(epochs,accuracy_averages)
    plt.title('Accuracy Vs # of Epochs')
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy')

    plt.savefig('Epoch_Accuracy.png')


def print_fold_accuracy(accuracies, folds):
    plt.figure(figsize=(9, 12))
    accuracy_averages = []
    for i in for_range(accuracies):
        accuracy_averages.append(np.average(accuracies[i]))
    plt.plot(folds, accuracy_averages)
    plt.title('Accuracy Vs # of Folds')
    plt.xlabel('# of Folds')
    plt.ylabel('Accuracy')

    plt.savefig('Fold_Accuracy.png')




def get_accuracies(data,attributes,num_epochs,num_folds,learning_rate,ROC = False):


    print "Data Loaded"

    #pdb.set_trace()


    print "Classes Renamed to 0 and 1"

    data = np.append(data,array([for_range(data)]).T,axis = 1)
    folded_data = folds(data,num_folds)
    #pdb.set_trace()
    net_shape = [size(data,1)-2,size(data,1)-2,1]

    print "Data Folded"

    netty = Net(net_shape)


    accuracies = zeros(num_folds)
    print_val = np.empty((size(data,axis = 0)),dtype = object)

    ROC_results = np.empty((0,4))

    for i in arange(num_folds):
        netty.random_weights(weight_std_dev=.1)
        results = np.empty((0,4))
        fold_indices = np.delete(arange(num_folds),i)
        current_train = [folded_data[iii] for iii in fold_indices]
        current_test = np.reshape(folded_data[i],(size(folded_data[i])/62,62))#TODO get rid of 61 as it is a magic number

        training_data = np.empty((0, size(current_train[0], axis=1)))
        for k in for_range(current_train, axis=0):
            training_data = np.concatenate((training_data, current_train[k]), axis=0)
        for j in arange(num_epochs):

                #pdb.set_trace()
            for k in training_data:
                #netty.propagate(j[:-1])
                correct = k[-2]
                test_num = k[-1]
                features = k[:-2]
                #results.append((correct,sigmoids[-1]))
                netty.back_propagate_2(correct,features)
        #print "Training Done"
        for k in current_test:
            correct = k[-2]
            test_num = k[-1]
            features = k[:-2]
            prediction = netty.propagate_2(features)
            confidence = confidence_of_prediction(prediction)
            results = np.append(results,[[int(test_num),int(correct),prediction,confidence]],axis = 0)
            if(ROC):
                ROC_results = np.append(ROC_results,[[int(test_num),int(correct),prediction,confidence]],axis = 0)
        #print "Fold %d Done!"%i

        accuracy,correct,total = 0,0,0
        for result in for_range(results):
            total+=1
            if(results[result][2]>=.5):
                results[result][2] = 1
            else:
                results[result][2] = 0
            if(results[result][1]==results[result][2]):
                correct += 1.0
        #print results
        accuracy = correct/total
        accuracies[i] = accuracy

        for result in results:
            ret_string = (str(i) + " " + attributes[-1][1][int(result[1])] + " " + attributes[-1][1][int(result[2])] + " " + str(result[3]))
            print_val[int(result[0])] = copy.deepcopy(ret_string)
            #print ret_string
            #print print_val[int(result[0])]
        #print "Accuracy: " + str(accuracy)
        accuracies[i] = accuracy

    if(ROC):
        return accuracies,ROC_results
    for i in print_val:
        print i
    return print_val

def confidence_of_prediction(predict_value):
    if predict_value <= .5:
        confidence = 1 - predict_value
    else:
        confidence = predict_value
    confidence -= .5
    confidence *= 2
    return confidence


def folds(data,num_folds,replacement = False):

    folds_final = [array([])]*num_folds

    data_a = data[data[:, -2] == 0]
    data_b = data[data[:, -2] == 1]

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


def sigmoid_prime(x):
    return exp(-x)/((1.0+exp(-x))**2)


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
        self.net_shape = net_shape
        self.random_weights(weight_std_dev)

        '''self.bias_weights = empty(size(net_shape, axis=0)-1, dtype=matrix)
                self.shape = net_shape
                for i in for_range(net_shape[:-1]):
                    rand_weights = np.random.normal(0.0,weight_std_dev,net_shape[i]*net_shape[i+1]) #TODO add these numbers as default arguments
                    rand_weights = reshape(rand_weights,(net_shape[i],net_shape[i+1]))
                    self.weights[i] = matrix(rand_weights,dtype = float)
                    rand_weights = np.random.normal(0.0, weight_std_dev, net_shape[i + 1])
                    self.bias_weights[i] = matrix(rand_weights,dtype = float)
        '''


    def random_weights(self,weight_std_dev = .001):

        self.W1 = np.random.normal(0.0, weight_std_dev, (self.net_shape[0] + 1, self.net_shape[1]))
        self.W2 = np.random.normal(0.0, weight_std_dev, (self.net_shape[1] + 1, self.net_shape[2]))

    def propagate_2(self,input_vars):

        input_vars = np.reshape(input_vars,(1,size(input_vars)))
        input_vars = np.concatenate((input_vars,[[1]]),1)
        self.hidden_vars = np.dot(input_vars,self.W1)
        self.hidden_sigmoids = sigmoid(self.hidden_vars)
        self.hidden_sigmoids = np.concatenate((self.hidden_sigmoids,[[1]]),1)
        self.output_vars = np.dot(self.hidden_sigmoids,self.W2)
        output = sigmoid(self.output_vars)
        return output


    def back_propagate_2(self,correct,input_vars,learning_rate = .1):
        old_w1 = copy.deepcopy(self.W1)
        old_w2 = copy.deepcopy(self.W2)


        output = self.propagate_2(input_vars)
        input_vars = np.reshape(input_vars, (1, size(input_vars)))
        #input_vars = np.concatenate((input_vars, [[1]]),1)


        delta = np.multiply(-(correct-output),sigmoid_prime(self.output_vars))
        delta_W2 = np.dot(self.hidden_sigmoids.T,delta)

        delta_h = np.dot(delta,self.W2[:-1].T)*sigmoid_prime(self.hidden_vars)
        delta_W1 = np.dot(input_vars.T,delta_h)

        self.W1[:-1] -= learning_rate * delta_W1
        self.W2 -= learning_rate * delta_W2

        return output

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





