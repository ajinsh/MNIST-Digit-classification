import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from helper_functions import *
from mlp import mlp

def plot_accuracy_graph(train_accuracy_matrix, test_accuracy_matrix):

    """
    This takes in the training accuracy and test accuracy as arguments and 
    plots the data onto an X-Y plane

    train_accuracy_matrix: all the training accuracies corresponding to each epoch

    test_accuracy_matirx: all the test accuracies corresponding to each epoch

    """

    train_epoch = train_accuracy_matrix[0]
    # print("train_epoch",train_epoch)
    train_accuracy = train_accuracy_matrix[1]
    plt.plot(train_epoch, train_accuracy, label = "Training Accuracy", linestyle = 'dashed')

    test_epoch = test_accuracy_matrix[0]
    test_accuracy = test_accuracy_matrix[1]
    plt.plot(test_epoch, test_accuracy, label = "Test Accuracy", linestyle = 'dashed')

    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.title("Experiment-3 \n epochs = 50 learning rate = 0.1  hidden_units = 100 \n momentum = 0.9")

    plt.legend()
    plt.show()


def main():

    """
    Read the data in the MNIST csv files using pandas read_csv and 
    convert each of them into numpy arrays

    Split both train and test numpy arrays to corresponding Y and X numpy arrays

    Normalize the X_train and X_test as the values range between 0 to 255

    add bias to the X_train and X_test

    One hot encode the Y_train and Y_test to make them compatible 
    to be compared to the output values of the neural network

    Train the model using the data based on different hyperparameters

    """

    data_train = pd.read_csv('MNIST_Dataset/mnist_train.csv')
    data_test= pd.read_csv('MNIST_Dataset/mnist_test.csv')

    data_train=np.array(data_train)
    data_test = np.array(data_test)

    rows_train_data, columns_train_data=data_train.shape
    rows_test_data, columns_test_data = data_test.shape

    Y_train = data_train[:,0]
    Y_test = data_test[:,0]

    X_train = data_train[:,1:]
    X_train = normalize_array(X_train)
    biased_X_train= add_bias(X_train)
    
    X_test = data_test[:,1:]
    X_test = normalize_array(X_test)
    biased_X_test = add_bias(X_test)

    Y_train_target = one_hot_encode_array(Y_train)
    Y_test_target = one_hot_encode_array(Y_test)


    print("Data Loaded..\n")
    print("No of Train Data samples : ", rows_train_data)
    print("Entering MLP\n\n\n")
    # print(Y_train_target[0].shape)
    print("As of now mlp function has not been implemented. It will be implemented in future iterations")

    hidden_units = 20
    train_accuracy_matrix, test_accuracy_matrix= mlp(biased_X_train, Y_train_target, biased_X_test, Y_test_target, 20, 50, 0.1, 0.9)
    
    # np.random.shuffle(data_train)

    # half_data_train = data_train[0:30000]
    # half_Y_train = half_data_train[:,0]
    # half_X_train = half_data_train[:,1:]

    # half_X_train = normalize_array(half_X_train)
    # biased_half_X_train = add_bias(half_X_train)

    # half_Y_train_target = one_hot_encode_array(half_Y_train)

    # train_accuracy_matrix, test_accuracy_matrix= mlp(biased_half_X_train, half_Y_train_target, biased_X_test, Y_test_target, 100, 5, 0.1, 0.9)

    # print(train_accuracy_matrix)
    # print(test_accuracy_matrix)
    # print("Success!!")
    plot_accuracy_graph(np.array(train_accuracy_matrix).T,np.array(test_accuracy_matrix).T)


if __name__=='__main__':
    main()