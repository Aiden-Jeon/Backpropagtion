import numpy as np
class Activation:
    def __init__(self,func):
        self.func = func

    def forward(self,x): 
        if self.func is 'sigmoid':
            return  1/(1+np.exp(-x))    
        if self.func is 'ReLU':
            x[x<0] = 0  
            return x
            
    def backward(self,x):
        if self.func is 'sigmoid':
            return  x*(1-x)
        if self.func is 'ReLU':
            x[x>=0] = 1
            x[x<0] = 0
            return x
            
class NeuralNetwork:
    def __init__(self,input_layer,hidden_layer,output_layer):
        self._o_layer = output_layer
        self.parms = {}
        self.parms['wih'] = np.random.normal(size=(input_layer,hidden_layer)) / np.sqrt(input_layer/2) 
        self.parms['who'] = np.random.normal(size=(hidden_layer,output_layer))/ np.sqrt(hidden_layer/2) 
        self.parms['b1'] = np.zeros(1)
        self.parms['b2'] = np.zeros(1)

    def _normalize(self,x): return (x-np.min(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0)+0.01)
    def _forward(self,x,b,w): return(b + x.dot(w))

    def train(self,x_train,y_train,lr=0.01,epoch=1,normalize=True,active_function='sigmoid',show_progress=True):
        self.y_train = np.zeros(shape=(y_train.shape[0],self._o_layer))
        self.y_train[np.arange(y_train.shape[0]),y_train] = 1
        self._target = y_train
        self._active = Activation(active_function)
        if normalize is True:
            self.x_train = self._normalize(x_train)
        else:
            self.x_train = x_train

        for ep in range(epoch):
            for idx in range(len(x_train)):
                #set parameters
                wih = self.parms['wih']
                who = self.parms['who']
                b1 = self.parms['b1']
                b2 = self.parms['b2']
                x_temp = np.array(self.x_train[idx],ndmin=2); y_temp = self.y_train[idx]
                
                hidden_out = self._active.forward(b1 + x_temp.dot(wih))
                output_out = self._active.forward(b2 + hidden_out.dot(who))
                output_par = np.array((output_out - y_temp)* self._active.backward(output_out),ndmin=2)
                hidden_par = np.array((output_par.dot(who.T) * self._active.backward(hidden_out)), ndmin=2)
                #update
                self.parms['who'] = self.parms['who'] - lr * np.array(hidden_out,ndmin=2).T.dot(output_par)
                self.parms['b2'] = self.parms['b2'] - lr * np.sum(output_par)
                self.parms['wih'] = self.parms['wih'] - lr * np.array(x_temp,ndmin=2).T.dot(hidden_par)
                self.parms['b1'] = self.parms['b1'] - lr * np.sum(hidden_par)
                
            if show_progress is True:
                if ep % 100 == 0 :
                    print(self.accuracy())
                    
    def accuracy(self):
        wih = self.parms['wih']
        who = self.parms['who']
        b1 = self.parms['b1']
        b2 = self.parms['b2']

        hidden_out = self._active.forward(b1 + self.x_train.dot(wih))
        output_out = self._active.forward(b2 + hidden_out.dot(who))
        pred = np.argmax(output_out,axis=1)
        self.pred = pred
        count=0
        for i in range(len(pred)):
            if pred[i]==self._target[i]:
                count += 1
        return count/len(pred)
