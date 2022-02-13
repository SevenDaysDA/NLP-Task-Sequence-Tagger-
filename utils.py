import json
import os


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def load_data(path):
    """ Load data from the given data set.
    Format [[Sentence, labels],
            [Sentence, labels]]

    Example:
    '''

    '''
    
    """
    with open(path + ".conll") as f:
        sentences = []
        labels = []
        s = []
        l = []
        for word in f.read().splitlines():
            if word.split():
                x,_,_,y = word.split()
                s.append(x)
                l.append(y)
            else:
                sentences.append(s.copy())
                labels.append(l.copy())
                s.clear()
                l.clear()
        data = [[sentences[i],labels[i]] for i in range(len(sentences))]
        return data


# Functions for Evaluation
#   P_micro = P_macro
def P_micro (C):
  diag = C.diagonal()
  return diag.sum() / C.sum()

def P_i (C,i):
  return C[i,i]/ C[:,i].sum()

def R_i (C,i):
  return C[i,i]/ C[i].sum()

def P_macro(C):
  sum_list = sum([P_i(C,i) for i in range(len(C))])
  return sum_list / len(C)

def R_macro(C):
  sum_list = sum([R_i(C,i) for i in range(len(C))])
  return sum_list / len(C)

def F1_score (C, variant):
  if variant == "macro":
    p_val = P_macro(C)
    r_val = R_macro(C)
    return (2*p_val*r_val) / (p_val+r_val)  # 2*P*R
  elif variant == "micro":
    return P_micro(C)
  else:
    print("Please define F1_score variant")



