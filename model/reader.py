from ctypes import sizeof
import numpy as np

import os

def load_data(filename, data_path="data"):

    print("Load Embedding")
    # Load Embedding
    embedding_dict = {}
    with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            key = line.split()[0]
            value = np.array(list(map(float,line.split()[1:51])))
            embedding_dict[key] = value
    print("--------------------------------------------")

    with open(os.path.join(data_path,filename), 'r') as lines:
            input, label = [] , []
            for line in lines:
                if line.strip():
                    x_val, irr1,irr2, y_val = line.split() 
                    input.append(x_val)
                    label.append(y_val)
            
            return input,label