import os
from model import bilstm


def run(params):
    print("Project")
    # set model output path
    params["model_path"] = os.path.join("results", "{}.model".format(params["model"]))

    # create model
    if params["model"] == "bilstm":
        model = bilstm.BiLSTM(params)
    else:
        raise ValueError("Unknown model: {}".format(params["model"]))

    # get predictions: list of tuples [(truth_1, pred_1), (truth_2, pred_2), ...]
    predictions = model.train_and_predict()


    print("Model has been trained and finished")
    # write predictions to file
    '''
    outfile = os.path.join("results", "{}_predictions.txt".format(params["model"]))
    with open(outfile, 'w') as file:
        file.write("\t\t".join(["input", "prediction", "truth"]) + "\n")
        file.write("-"*35+"\n")
        for line in predictions:
            file.write("\t\t".join(line) + "\n")
    '''
    
    # print word accuracy
    '''
    tp = sum([1 if t == p else 0 for _,p,t in predictions])
    acc = tp / len(predictions)
    print("Word test accuracy of {}: {}".format(params["model"], acc))
    '''


if __name__=='__main__':
    params = {"model": "bilstm",   
              "batch_size": 1,
              "dropout": 0.5,
              "hidden_units": 100,
              "epochs": 1}
    run(params)