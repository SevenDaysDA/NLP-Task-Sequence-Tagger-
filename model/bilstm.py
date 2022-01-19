from importlib.resources import path
import os
from unittest import result
import numpy as np
np.random.seed(42)
import math

from reader import load_data




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
        X_TEST_raw , Y_TEST_raw = [],[]
        X_DEV_raw , Y_DEV_raw = [],[]
        X_TRAIN_raw , Y_TRAIN_raw = [], []


    def load_data(self, pathname):
        print("Load Data ")
        # Load data:
        print(os.getcwd())
        print("none")
        X_TEST_raw , Y_TEST_raw = load_data(filename="test.conll",data_path=pathname+"data")
        X_DEV_raw , Y_DEV_raw = load_data("dev.conll")
        X_TRAIN_raw , Y_TRAIN_raw = load_data("train.conll")
        print("--------------------------------------------")
        print("change")
        print("Load Embedding")
        # Load Embedding
        embedding_dict = {}

        with open(".glove.6B.50d.txt", 'r', encoding="utf-8") as f:
            for line in f:
                key = line.split()[0]
                value = np.array(list(map(float,line.split()[1:51])))
                embedding_dict[key] = value
        print("--------------------------------------------")


        

    def train_and_predict(self):
        hidden_units = self.params["hidden_units"]
        dropout_drop_prob = self.params["dropout"]
        batch_size = self.params["batch_size"]
        model_path = self.params["model_path"]


        
        #[[char_to_index[char] for char in word] for word in X_train_data]

        X_TEST_embed = [[embedding_dict[key] for key in word] for word in X_TEST_raw]
        print(len(X_TEST_raw))
        print(len(X_TEST_embed))



        result  = "hi"
        return result