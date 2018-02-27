import numpy as np
class activation:
        def forward(self,x,method='sigmoid'):
            x = np.array(x)
            if method.lower() == 'sigmoid':
                return 1/(1+np.exp(-x))
            if method.lower() == 'relu':
                x[x<0] = 0 
                return x

        def backward(self,x,method='sigmoid'):
            x = np.array(x)
            if method.lower() == 'sigmoid':
                return x*(1-x)
            if method.lower() == 'relu':
                x[x<0] = 0 
                x[x>0] = 1
                return x
##
class layers:
    def weight(self,shape=[None,None]):
        return np.random.normal(size = shape) / np.sqrt(shape[0]/2)

    def layer(self,X,W,b,active='sigmoid'):
        if type(X) is dict:
            X_t = X['out']
            out = activation().forward(X_t.dot(W)+b,active)
            out_back = activation().backward(out,active)
            return dict(out=out, out_back=out_back, weight=W, bias=b, active=active, net= X_t)
        else:
            X_t = np.array(X)
            out = np.array(X_t).dot(W)+b
            out = activation().forward(out,active)
            out_back = activation().backward(out,active)
            return dict(out=out, out_back=out_back, weight=W, bias=b, active=active, net= X_t)
    def train_data(self,shape=[None,None]):
        return np.zeros(shape[1])

##
class optimizer:
    def loss(self,real,predict,loss_type='mse'):
        if loss_type.lower() == 'mse':
            return np.array(predict) - np.array(real)
        if loss_type.lower() == 'cross_entropy':
            return np.array(real) * np.array(predict)
        
    def accuracy(self,pred,real):
        count = 0 
        pred = np.argmax(pred,axis=1)
        real = np.where(real==1)[1]
        for i in range(len(pred)):
            if pred[i]==real[i]:
                count += 1
        return count/len(pred)

    def SGD(self,x,y,layer,loss_type,lr=0.01,epoch=10):
        def layer_cal(train_data,weight,bias):
            layer[0] = layers().layer(train_data,weight[0],bias[0])
            for la in range(1,len(weight)):
                layer[la] = layers().layer(layer[la-1]['out'],weight[la],bias[la])
            return layer
        
        def dictionary(layer):
            out = dict();out_back = dict();weight = dict();bias = dict();active = dict();net = dict()
            for i in range(len(layer)):
                out[i]    = layer[i]['out']
                out_back[i] = layer[i]['out_back']
                weight[i] = layer[i]['weight']
                bias[i]   = np.array(layer[i]['bias'])
                active[i] = layer[i]['active']
                net[i]    = np.array(layer[i]['net'])
            return out,out_back,weight,bias,active,net
        
        def out_net_cal(out,out_back):
            out_net = dict()
            for i in reversed(range(len(out))):
                out_net_temp = np.eye(out[i].shape[0])
                np.fill_diagonal(out_net_temp,out_back[i])
                out_net[i] = out_net_temp
            return out_net
        
        def training(loss_val,out_net,out_back,weight,bias):
            w_update = dict(); b_update=dict()
            for idx in reversed(range(len(weight))):
                update = np.array(loss_val)
                for i in reversed(range(idx,len(weight))):
                    if i == idx:
                        b_upd = (update.dot(out_net[i]))
                        update = update.dot(out_net[i])
                        update = net[i].reshape(net[i].shape[0],1).dot(update.reshape(1,update.shape[0]))
                    else:
                        update = (update.dot(out_net[i])).dot(weight[i].T)
                w_update[idx] = update
                b_update[idx] = b_upd
            return w_update, b_update
        
        ## 전체 data set 
        out,out_back,weight,bias,active,net = dictionary(layer)
        for ep in range(epoch):
            for iteration in range(len(x)):
                layer = layer_cal(x[iteration],weight,bias)
                out,out_back,_,_,active,net = dictionary(layer)
                out_net = out_net_cal(out,out_back)
                loss_val = self.loss(y[iteration],out[len(out)-1],loss_type)
                w_update, b_update = training(loss_val,out_net,out_back,weight,bias)
                ## update
                for i in range(len(weight)):
                    weight[i] = weight[i] - lr*w_update[i]
                    bias[i] = bias[i] - lr*b_update[i]

            if ep % 10 == 0:
                temp = activation().forward(np.array(x).dot(weight[0])+bias[0],active[0])
                for i in range(1,len(weight)):
                    temp =  activation().forward(temp.dot(weight[i])+bias[i],active[i])
                print(self.accuracy(temp,y))
                
        return dict(weight=weight,bias=bias)

#-------------------------------------------------
#example
W1 = layers().weight([3,4])
W2 = layers().weight([4,2])
W3 = layers().weight([2,3])
x_t=[[1, 2, 3], [3, 4, 5], [6, 7, 8]]
x_data = layers().train_data([3,3])
layer1 = layers().layer(x_data,W1,[1,1,1,2])
layer2 = layers().layer(layer1,W2,[1,2])
layer3 = layers().layer(layer2,W3,[1,2,1])
layer = [layer1,layer2,layer3]
y= np.array([[1,0,0],[0,1,0],[0,0,1]])
optimizer().SGD(x_t,y,layer,lr=0.01,loss_type='mse',epoch=10)
