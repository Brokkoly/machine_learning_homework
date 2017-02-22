


import numpy as np

class Priority_Queue:



    def __init__(self):
        self.data = np.array([])


    def put(self,newData):
        self.data = np.append(self.data,newData)
        self.data = np.reshape(self.data,(np.size(self.data)/2,2))

    def get(self):
        index = -1
        maxPriority = -1
        for i in np.arange(np.size(self.data,axis = 0)):
            if(self.data[i,0] > maxPriority):
                maxPriority = self.data[i,0]
                index = i
        retval = self.data[index]
        self.data = np.delete(self.data,index)
        self.data = np.delete(self.data,index)
        self.data = np.reshape(self.data,(np.size(self.data)/2,2))
        return retval

    def modify_priority(self,value,priority):
        for i in np.arange(np.size(self.data)):
            if(self.data[i,1]==value):
                self.data[i,0]=priority
                break

    def empty(self):
        if(self.size()==0):
            return True
        else:
            return False

    def size(self):
        return np.size(self.data,axis = 0)

    def contains(self,value):
        for i in np.arange(np.size(self.data,axis = 0)):
            datacheck = self.data[i,1]

            if(self.data[i,1] == value):
                return True
        return False