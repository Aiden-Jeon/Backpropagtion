import numpy as np
import activation
import layers
#define loss type
def _loss(real,predict,loss_type='mse'):
    if loss_type.lower() == 'mse':
        return np.array(predict) - np.array(real)
    if loss_type.lower() == 'cross_entropy':
        return -np.array(1/(predict))*np.array(real) + np.array(1/(1-predict))*(1-np.array(real))

#accuracy [predict, real value]
def accuracy(pred,real):
    count = 0 
    pred = np.argmax(pred,axis=1)
    real = np.where(real==1)[1]
    for i in range(len(pred)):
        if pred[i]==real[i]:
            count += 1
    return count/len(pred)

#feeing forward
def _forward(layer,train_data,weight,bias):
    layer[0] = layers.layer(train_data,weight[0],bias[0])
    for la in range(1,len(weight)):
        layer[la] = layers.layer(layer[la-1]['out'],weight[la],bias[la])
    return layer

# making dictionart before training
def _dictionary(layer):
    out = dict();out_back = dict();weight = dict();bias = dict();active = dict();net = dict()
    for i in range(len(layer)):
        out[i]    = layer[i]['out']
        out_back[i] = layer[i]['out_back']
        weight[i] = layer[i]['weight']
        bias[i]   = np.array(layer[i]['bias'])
        active[i] = layer[i]['active']
        net[i]    = np.array(layer[i]['net'])
    return out,out_back,weight,bias,active,net


def _out_net_cal(out,out_back):
    out_net = dict()
    for i in reversed(range(len(out))):
        out_net_temp = np.eye(out[i].shape[0])
        np.fill_diagonal(out_net_temp,out_back[i])
        out_net[i] = out_net_temp
    return out_net

def _train(loss_val,out_net,out_back,net,weight,bias):
    w_update = dict(); b_update=dict()
    for idx in reversed(range(len(weight))):
        update = np.array(loss_val)
        for i in reversed(range(idx,len(weight))):
            if i == idx:
                b_upd  = update.dot(out_net[i])
                update = update.dot(out_net[i])
                update = net[i].reshape(net[i].shape[0],1).dot(update.reshape(1,update.shape[0]))
            else:
                update = (update.dot(out_net[i])).dot(weight[i].T)
        w_update[idx] = update
        b_update[idx] = b_upd
    return w_update, b_update


def GradientDescent(x,y,layer,loss_type,lr=0.01,epoch=10):
    out,out_back,weight,bias,active,net = _dictionary(layer)
    for ep in range(epoch):
        w_log = dict(); b_log = dict()
        for i in range(len(weight)):
            w_log[i] = weight[i] * 0.0
            b_log[i] = bias[i] * 0.0
        for iteration in range(len(x)):
            layer = _forward(layer,x[iteration],weight,bias)
            out,out_back,_,_,active,net = _dictionary(layer)
            out_net = _out_net_cal(out,out_back)
            loss_val = _loss(y[iteration],out[len(out)-1],loss_type)
            w_update, b_update = _train(loss_val,out_net,out_back,net,weight,bias)
            for i in range(len(weight)):
                w_log[i] += w_update[i]
                b_log[i] += b_update[i]
        for i in range(len(weight)):
            w_log[i] = w_log[i]/(len(x)+1)
            weight[i] = weight[i] - lr*w_log[i]
            b_log[i] = b_log[i]/(len(x)+1)
            bias[i] = bias[i] - lr*b_log[i]
        
        if ep % 1000 == 0:
            temp = activation.forward(np.array(x).dot(weight[0])+bias[0],active[0])
            for i in range(1,len(weight)):
                temp =  activation.forward(temp.dot(weight[i])+bias[i],active[i])
            print(accuracy(temp,y))
            
    return dict(weight=weight,bias=bias)