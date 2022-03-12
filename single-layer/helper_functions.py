import numpy as np


def normalize_array(array):

    """
    normalizes all the values in the input array

    Arguments:

    array -- This array refers to the input array that has to be normalized
    
    """

    normalized_array=np.true_divide(array, 255)

    return normalized_array

def add_bias(array):

    """
    adds bias as a column to that input array

    Arguments:

    array -- This array refers to the input array to which bias has to be added
    
    """
        
    bias_matrix=np.ones(np.shape(array)[0])
    array=np.insert(array, 0, bias_matrix, axis=1)

    return array

def one_hot_encode_array(array):

    """
    Returns one-hot encoded array based on the number of unique classes the input array possesses

    Arguments:
    
    array -- This array refers to the input array which has to be one-hot encoded

    """

    no_of_classes=len(np.unique(array))
    one_hot_encoded_array=np.eye(no_of_classes)[array.reshape(-1)]
    # one_hot_encoded_array= np.where(one_hot_encoded_array==1, 0.9, 0.1)

    return one_hot_encoded_array

def reshape_matrix(array, new_order):

    """
    reshapes the input array to a desired order

    Arguments:

    array -- This array refers to the input array which has to be reshaped

    new_order -- The order of the desired matrix

    """

    reshaped_array = np.reshape(array, new_order)

    return reshaped_array

def initialize_weight_matrix(no_of_units_to_be_activated, no_of_inputs, flag=1, flag_w2=0):

    """
    Returns a matrix filled with randomly generated values ranging between -0.05 and 0.05

    Arguments:

    no_of_units_to_be_activated -- This refers to the no of units in the hidden or output layer that need to be activated

    no_of_inputs -- This referes to the no of units in the layer that is activating the hidden or output layer

    """

    # if(flag==0):
    # input_array=reshape_matrix(input_array, (1, input_array.shape[0]))
    if(flag_w2==1):
        weight_matrix = np.random.uniform(-0.05, 0.05, (no_of_units_to_be_activated, no_of_inputs+1))    
    else:
        weight_matrix = np.random.uniform(-0.05, 0.05, (no_of_units_to_be_activated, no_of_inputs+1))

    return weight_matrix