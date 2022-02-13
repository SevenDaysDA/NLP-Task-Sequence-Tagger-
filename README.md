## Usage

1. **Prepare** Glove-Embedding unzipping embedding-file and placing it in this file.

2.  **Set** Parameters for learning, which are saved in 'params.json'.

3. **Train** experiments by running
```
python SeqTagger.py
```

After the script is done with the training youe will find the raw scores in 'results/Scores_raw.txt'

4. **Visualize** the F1 scores and the loss with:
```
python Visualize.py
```