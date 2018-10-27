import csv
import numpy as np
import math
from scipy.special import expit
from sklearn.metrics import confusion_matrix, accuracy_score
import random


f = open("mnist_train.csv","r")
data0 = csv.reader(f)
train = np.array(list(data0))


f1 = open("mnist_test.csv","r")
data1 = csv.reader(f1)
test = np.array(list(data1))


### experiment 3 reshuffle data and select 1/4 or 1/2

np.random.shuffle(train)
train = train[0:15000]
# print(train.shape)

bias = 1
lear_rate = 0.1
alpha = 0.9

# No. of hidden inputs
n = 100


weight_i2h = np.random.uniform(-0.05,0.05,(785,n))
# print(weight_i2h.shape)

weight_h20 = np.random.uniform(-0.05,0.05,(n+1,10))
# print(weight_h20.shape)

# store previous delta wt from hidden to output layer
prev_wt_h20 = np.zeros((n+1,10))

# store previous delta wt from input to hidden layer
prev_wt_i2h = np.zeros((785,n))


# matrix to store the activation h1...hk 
hl_input = np.zeros((1,n+1))
hl_input[0,0] = 1


# print(hl_input.shape)






def multi_perceptron(epoch,input_ds,set_flag):
    global weight_i2h,weight_h20,prev_wt_i2h,prev_wt_h20
    pred_list = []
    actual_list = []
    for i in range(input_ds.shape[0]):
        target_class = input_ds[i,0].astype('int')
        actual_list.append(target_class)    
        xi = input_ds[i].astype('float16')/255
        xi[0] = bias            ## Set the value of x0 to bias unit = 1
        xi = xi.reshape(1,785)

        z_hl = np.dot(xi,weight_i2h)
        sig_hl = expit(z_hl)
        # print("sig_hl",sig_hl.shape)
        hl_input[0,1:] = sig_hl
        # print("hl_input",hl_input)
        # print(hl_input.shape)
        z_ol = np.dot(hl_input,weight_h20)
        sig_ol = expit(z_ol)
        # print(sig_ol)
        predict = np.argmax(sig_ol)
        # print(predict)
        pred_list.append(predict)
        # print(type(sig_ol))
        # print(sig_ol.shape)


        if epoch>0 and set_flag == 1:
            # print("inside wt updation",epoch)
            ###### Calculating error term #######

            ##error term for output unit 
            tk = np.zeros((1,10))+0.1
            tk[0,target_class] = 0.9
            # print(tk)
            error_ol = sig_ol*(1-sig_ol)* (tk - sig_ol)
            # print("error_ol shape for ",epoch," ",error_ol.shape)
            ##error term for hidden unit
            error_hl = sig_hl*(1-sig_hl)*np.dot(error_ol,weight_h20[1:,:].T) 
            # print(delta_hl.shape)
            # print("error_hl shape for ",epoch," ",error_hl.shape)
            ####### Update weights ##########

            ### Hidden to output layer wt updation

            delta_weight_h20 = (lear_rate * error_ol * hl_input.T) + (alpha * prev_wt_h20)
            prev_wt_h20 = delta_weight_h20
            # print("delta_weight_h20.shape after wt updation", delta_weight_h20.shape)
            weight_h20 = weight_h20 + delta_weight_h20

            ### Input to output layer wt updation    

            delta_weight_i2h = (lear_rate * error_hl * xi.T) + (alpha * prev_wt_i2h) 
            prev_wt_i2h = delta_weight_i2h
            # print("delta_weight_i2h.shape after wt updation", delta_weight_i2h.shape)
            weight_i2h = weight_i2h + delta_weight_i2h



    accur = (np.array(pred_list) == np.array(actual_list)).sum()/float(len(actual_list))*100
        
    print("len of actual_list after ", epoch," is ",len(actual_list))
    print("len of pred_list after ", epoch," is ",len(pred_list))

    if(set_flag == 0):
        print("Confusion matrix for epoch ",epoch)
        print(confusion_matrix(actual_list,pred_list))  
    return accur


def store_accur(accur_index,accur,input_ds):
    with open(input_ds, 'a', newline='') as myfile:
     wr = csv.writer(myfile)
     wr.writerow([accur_index,accur])



for each in range(50):
    trn_accuracy = multi_perceptron(each,train,1)
    tst_accuracy = multi_perceptron(each,test,0)
    store_accur(each,trn_accuracy,'train_quarter_α='+str(alpha)+'.csv')
    store_accur(each,tst_accuracy,'test_quarter_α='+str(alpha)+'.csv')
