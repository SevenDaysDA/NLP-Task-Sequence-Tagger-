from ctypes import sizeof
import numpy as np

import os

def load_data(filename, data_path=".data"):

    with open(os.path.join(data_path,filename), 'r') as lines:
            input, label = [] , []
            for line in lines:
                if line.strip():
                    x_val, irr1,irr2, y_val = line.split() 
                    input.append(x_val)
                    label.append(y_val)
            
            return input,label