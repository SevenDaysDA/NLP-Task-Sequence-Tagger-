import os
from unittest import result
import numpy as np
np.random.seed(42)
import math

from .reader import load_data




def get_index_dict(input_data):
    """
    Create index - word/label dict from list input
    @params : List of lists
    @returns : Index - Word/label dictionary
    """
    result = dict()
    vocab = set()
    i = 1
    # Flatten list and get indices
    for element in [word for sentence in input_data for word in sentence]:
        if element not in vocab:
            result[i]=element
            i+=1
            vocab.add(element)
    return result

class BiLSTM():

    def __init__(self, params):
        self.params = params



        

    def train_and_predict(self):
        hidden_units = self.params["hidden_units"]
        dropout_drop_prob = self.params["dropout"]
        batch_size = self.params["batch_size"]
        model_path = self.params["model_path"]

        # Load data:
        X_TEST , Y_TEST = load_data("test.conll")
        X_DEV , Y_DEV = load_data("dev.conll")
        X_TRAIN , Y_TRAIN = load_data("train.conll")


        print(len(X_TRAIN))
        print(len(Y_TRAIN))
        print(len(X_DEV))
        print(len(Y_DEV))



        result  = "hi"
        return result