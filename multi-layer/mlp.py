from helper_functions import *

def mlp_test(biased_X_test, Y_test_target, W1, W2):
    
    """
    Computes and returns the outputs for the test data set

    Arguments:

    biased_X_test -- one sample of the test dataset

    Y_test_target -- actual output value of the above mentioned sample

    W1 -- weight matrix associated between input and hidden layer

    W2 -- weight matrix associated between hidden and output layer

    
    """
    
    Z1 = np.dot(biased_X_test, W1.T).astype('float32')
    A1 = Sigmoid(Z1) #shape - (10,)
    A1 = reshape_matrix(A1, (1, A1.shape[0])) #shape -(1,10)
    hidden_layer_input = add_bias(A1) #shape - (1,11)
    Z2 = np.dot(hidden_layer_input, W2.T).astype('float32') #shape - (1,10)
    A2 = Sigmoid(Z2) #shape - (1,10)

    return A2

def mlp(biased_X_train, Y_train_target, biased_X_test, Y_test_target,hidden_units, epochs, learning_rate, momentum):
    test_accuracy_matrix = list()
    train_accuracy_matrix = list()

    W1 = initialize_weight_matrix(hidden_units, 784) #shape - (10, 785)
    W2 = initialize_weight_matrix(10, hidden_units) #shape - (10,11)
    
    for epoch in range(0, epochs):
        print("Epoch : ", epoch)
        train_pred_list = list()
        test_pred_list = list()
        # print("length of pred train : ", len(train_pred_list))
        # print("length of pred test : ", len(test_pred_list))
        for row in range(0, biased_X_train.shape[0]):
            # print(j)
            if(epoch==0 and row==0):

                Z1 = np.dot(biased_X_train[row], W1.T).astype('float32') #shape - (10,)
                A1 = Sigmoid(Z1) #shape - (10,)
                A1 = reshape_matrix(A1, (1, A1.shape[0])) #shape -(1,10)
                hidden_layer_input = add_bias(A1) #shape - (1,11)

                Z2 = np.dot(hidden_layer_input, W2.T).astype('float32') #shape - (1,10)
                A2 = Sigmoid(Z2) #shape - (1,10)
                # print(A1)
                # print(A2)
                # # print(W1[0:3])
                # print(W2[0:3])

                train_prediction = np.argmax(A2)
                train_pred_list.append(train_prediction==np.argmax(Y_train_target[row]))

                if(row < biased_X_test.shape[0]):
                    test_output = mlp_test(biased_X_test[row], Y_test_target[row], W1, W2)
                    test_prediction = np.argmax(test_output)
                    test_pred_list.append(test_prediction==np.argmax(Y_test_target[row]))

                error_at_output_layer, error_at_hidden_layer = calculate_error_terms(hidden_layer_input, A2, W1, W2, Y_train_target[row]) 
                # eaol shape - (1,10)
                # eahl shape - (1,11)
                # print(error_at_output_layer)
                # print(error_at_hidden_layer)
                
                dW2, W2 = update_weights(W2, learning_rate, error_at_output_layer, hidden_layer_input, flag = 0)
                last_dW2_change = dW2
                
                dW1, W1 = update_weights(W1, learning_rate, error_at_hidden_layer[:,1:], biased_X_train[row], flag = 0)
                last_dW1_change = dW1

            else:
                # print(W1[0:3])
                # print(W2[0:3])
                # # break
                Z1 = np.dot(biased_X_train[row], W1.T).astype('float32') #shape - (10,)
                A1 = Sigmoid(Z1) #shape - (10,)
                A1 = reshape_matrix(A1, (1, A1.shape[0])) #shape -(1,10)
                hidden_layer_input = add_bias(A1) #shape - (1,11)
                Z2 = np.dot(hidden_layer_input, W2.T).astype('float32') #shape - (1,10)
                A2 = Sigmoid(Z2) #shape - (1,10)

                # print(A1)
                # print(A2)


                train_prediction = np.argmax(A2)
                train_pred_list.append(train_prediction==np.argmax(Y_train_target[row]))

                if(row < biased_X_test.shape[0]):
                    test_output = mlp_test(biased_X_test[row], Y_test_target[row], W1, W2)
                    test_prediction = np.argmax(test_output)
                    test_pred_list.append(test_prediction==np.argmax(Y_test_target[row]))

                error_at_output_layer, error_at_hidden_layer = calculate_error_terms(hidden_layer_input, A2, W1, W2, Y_train_target[row]) 

                # print(error_at_output_layer)
                # print(error_at_hidden_layer)

                dW2, W2 = update_weights(W2, learning_rate, error_at_output_layer, hidden_layer_input, momentum, last_dW2_change)
                last_dW2_change = dW2
                
                dW1, W1 = update_weights(W1, learning_rate, error_at_hidden_layer[:,1:], biased_X_train[row], momentum, last_dW1_change)
                last_dW1_change = dW1

        
        test_accuracy = np.mean(test_pred_list)
        train_accuracy = np.mean(train_pred_list)
        
        print("Epoch : ", epoch)
        print("Training accuracy : ", train_accuracy)
        print("Test accuracy : ", test_accuracy)
        
        train_accuracy_matrix.append([epoch, train_accuracy])
        test_accuracy_matrix.append([epoch, test_accuracy])

    # confusion_matrix = confusion_matrix(test_pred_list, np.argmax())

    return train_accuracy_matrix, test_accuracy_matrix


