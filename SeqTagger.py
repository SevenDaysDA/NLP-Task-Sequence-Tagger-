from itertools import permutations
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import utils
torch.manual_seed(1)


# Use of Pytorch and structure of model was inspired from 
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html 

# Prepare data:
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

######################################################################
# Create the model:
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional = True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# Evaluate Scores
def evaluate(loss, data_set = "dev"):
    print("Scores: ")
    y_truth = torch.empty(0)
    y_pred = torch.empty(0)
    # Create prediction and compare with truth labels
    if data_set == "test":
        eval_data = test_data
    else:
        eval_data = dev_data

    for iter in range(len(eval_data)): 
        inputs = prepare_sequence(eval_data[iter][0], word_to_ix)
        tag_scores = model(inputs)
        predictions = np.argmax(tag_scores,axis=1)
        # Add to final score array
        y_truth = torch.cat((y_truth,prepare_sequence(eval_data[iter][1], tag_to_ix).clone()))
        y_pred = torch.cat((y_pred,predictions.clone()))
    conf_mat = np.zeros((len(tag_to_ix),len(tag_to_ix)))            # conf_mat -> Confusion matrix 
    for i in range(len(y_truth)):
        conf_mat[int(y_truth[i].item()),int(y_pred[i].item())] += 1
    micro = utils.F1_score(conf_mat,"micro")
    macro = utils.F1_score(conf_mat,"macro")
    print("Micro F1:  ",micro)
    print("Macro F1:  ",macro)
    mic_list.append(micro)
    mac_list.append(macro)
    loss_hist.append(loss)


######################################################################
# Train the model:
def train_and_eval(params, model, loss_function, optimizer):

    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)


    for epoch in range(params.epochs): 
        print("start epoch ",epoch)
        loss_sum = 0

        permutation = torch.randperm(len(training_data))

        for i in range(0, len(training_data), params.batch_size):
            model.zero_grad()
            # Batch
            indices = permutation[i:i+params.batch_size]
            for j in indices:
                sentence , tags = training_data[j]
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = prepare_sequence(tags, tag_to_ix)
                tag_scores = model(sentence_in)
                loss = loss_function(tag_scores, targets)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            evaluate(loss_sum/len(training_data),"dev")
        print("end epoch")

    # See what the scores are after training
    with torch.no_grad():
        # Final Evaluation
        print("Final scores: ")
        truth_arr = torch.empty(0)
        predi_arr = torch.empty(0)
        for iter in range(len(test_data)):
            inputs = prepare_sequence(test_data[iter][0], word_to_ix)
            tag_scores = model(inputs)
            predictions = np.argmax(tag_scores,axis=1)
            # Add to final score array
            truth_arr = torch.cat((truth_arr,prepare_sequence(test_data[iter][1], tag_to_ix).clone()))
            predi_arr = torch.cat((predi_arr,predictions.clone()))
        evaluate(loss_sum/len(training_data),"test")

if __name__ == '__main__':

    # Embedding Preparation:
    vocab,embeddings = [],[]
    with open('glove.6B.50d.txt','rt',encoding = "utf-8") as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)
    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.
    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
    ###############################################################    
    # ->  Code Snippet for working with Glove-Embeddings (from https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a)


    # Load Data 
    # Format 
    training_data = utils.load_data("data/train")
    test_data = utils.load_data("data/test")
    dev_data = utils.load_data("data/dev")
    all_data=[training_data, test_data, dev_data]

    word_to_ix = {}
    # For each words-list (sentence) and tags-list in each tuple of training_data
    for data in all_data:
        for sent, tags in data:
            for word in sent:
                if word not in word_to_ix:  # word has not been assigned an index yet
                    word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index

    # Assign indexes for tags                
    tag_to_ix = {"O":0,"B-ORG":1,"B-MISC":2,"B-PER":3,"I-PER":4,"B-LOC":5,"I-ORG":6,"I-MISC":7,"I-LOC":8}
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}

    # Load Paramters from json
    json_path = 'params.json'
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

     # History arrays
    mic_list = []
    mac_list = []
    loss_hist = []

    # Create instance of model
    model = LSTMTagger(params.embedding_dim, params.hidden_dim, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    #Training
    train_and_eval(params, model, loss_function, optimizer)


    # Save Scores 
    with open("results/Scores_raw.txt", "w+") as w:
        w.write("F1_Macro" + "\t\t" +  "F1_Micro" + "\t\t\t" +  "Loss" + "\n")
        for i in range(len(mic_list)):
            w.write(str(mic_list[i]) + "\t" +  str(mac_list[i]) + "\t" +  str(loss_hist[i]) + "\n")








