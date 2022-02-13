import matplotlib.pyplot as plt
import numpy as np

macro, micro, loss = [] , [] , []
with open("results/Scores_raw.txt", "r") as f:
    for line in f.read().splitlines():
        values = line.split("\t")
        if "Macro" in  values[0]:
            continue
        else:
            macro.append(float(values[0])) 
            micro.append(float(values[1]))
            loss.append(float(values[2]))

plt.xlabel("Epochs")
plt.ylabel("Score")
plt.ylim([0,1])

plt.xticks(np.arange(0,len(macro),1))


plt.plot(micro, label = "F1 Micro" )
plt.plot(macro, label = "F1 Macro")
plt.plot(loss, label = "Loss")
plt.title("Scores")
plt.legend()
plt.savefig("results/Scores.png")
plt.show()

