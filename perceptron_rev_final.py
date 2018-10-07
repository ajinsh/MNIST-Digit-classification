
#### Author : Ajinkya Shinde


#### Load all the required packages###
import csv
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



## Load the train and test csv files into train_data and test_ data
## as numpy array
f = open('mnist_train.csv','r')
data = csv.reader(f)
list_data = list(data)

train_data = np.array(list_data)

f1 = open('mnist_test.csv','r')
data1 = csv.reader(f1)
list_data1 = list(data1)
test_data = np.array(list_data1)


### Initialize the weight variable of the dimension 10 x 785  where
### each single row of 1 x 785 is input to one single perceptron for
### a given training example
weight_arr = np.random.uniform(-0.05,0.05,(10,785))

### Set bias unit to one
bias = 1

### train_accuracy and test_accuracy are used to store the accuracy rates
### for training data and test data for each single epoch
train_accuracy = []
test_accuracy = []
lear_rate = 0.001
# lear_rate = 0.01
# lear_rate = 0.1

### perceptron_learn function : This function is called to train the perceptrons 
### and weight updation purposes. The first for loop iterates through each training
### example and the first inner for loop is used to calculate the max w.x i.e the 
### prediction for a single training example for a group of 10 perceptrons. The
### second inner for loop does the weight updation for all perceptrons.
### Parameters : 
###         epoch    :int   - to run through dataset 50 times
###         input_ds :array - to pass the dataset name - train/test as numpy array
###         set_flag :int   - used for not updating weights for test dataset
def perceptron_learn(epoch,input_ds,set_flag):
    pred_list = []
    actual_list = []
    for i in range(0,input_ds.shape[0]):
        target_class = input_ds[i,0].astype('int')
        target_list = [0,0,0,0,0,0,0,0,0,0]
        target_list[target_class] = 1
        
        xi = input_ds[i].astype('float16')/255
        xi[0] = bias            ## Set the value of x0 to bias unit = 1
        xi = xi.reshape(1,785)
        preact_list = []
        y_list = []
        actual_list.append(target_class)

        for p in range(10):
            preact =np.inner(xi,weight_arr[p,:])
            if(preact <= 0):
                prediction = 0
            else:
                prediction = 1

            preact_list.append(preact)
            y_list.append(prediction)
        

        preact_arr = np.array(preact_list)
        pred_list.append(np.argmax(preact_arr))
        if epoch > 0 and set_flag == 1:
            for q in range(10):
                weight_arr[q,:] = weight_arr[q,:] + (lear_rate * (target_list[q] - y_list[q]) * xi)
    accur = (np.array(pred_list) == np.array(actual_list)).sum()/float(len(actual_list))*100 


    if set_flag == 0:
        print("Confusion matrix for test data for epoch ",epoch)
        print(confusion_matrix(actual_list,pred_list))       
    return accur     


### store_accur function: used to store accuracy for each learning rate for either test/train dataset
### into respective csv files.
### Parameters:
###         accur_index : int       - calculated accuracy index for indicating the epoch no.
###         accur       : int       - calculated accuracy
###         input_ds    : string    - the file name with which to store the file with
def store_accur(accur_index,accur,input_ds):
    with open(input_ds, 'a', newline='') as myfile:
     wr = csv.writer(myfile)
     wr.writerow([accur_index,accur])

### Loop through training and test data for 50 epochs. calculate the accuracy and pass it to store_accur
### function         

for each in range(50):
    trn_accuracy = perceptron_learn(each,train_data,1)
    train_accuracy.append(trn_accuracy)
    tst_accuracy = perceptron_learn(each,test_data,0)
    test_accuracy.append(tst_accuracy)
    store_accur(each,trn_accuracy,'train_output_'+str(lear_rate)+'.csv')
    store_accur(each,tst_accuracy,'test_ouput'+str(lear_rate)+'.csv')




