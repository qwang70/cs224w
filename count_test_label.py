import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import pandas as pd

sns.set()
LabelName = {1: "Danaus plexippus", 2:"Heliconius charitonius", 3:"Heliconius erato", \
                4:"Junonia coenia", 5:"Lycaena phlaeas", 6:"Nymphalis antiopa", 7:"Papilio cresphontes",\
                8:"Pieris rapae", 9:"Vanessa atalanta", 10:"Vanessa cardui"}

with open("test_label_butterfly_emb_1_1.pkl", 'rb') as f:
    label_w = pkl.load(f)
arr = np.zeros((10,10))
for val in label_w.values():
    arr[val] += 1
arr_3col = []
for i in range(10):
    num = np.sum(arr[i])
    for j in range(10):
        arr_3col.append([LabelName[i+1], LabelName[j+1], arr[i,j]/num])
df = pd.DataFrame(arr_3col,columns=['predicted label', 'actual label', 'precision'])

f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df.pivot('predicted label', 'actual label', 'precision'), annot=True, linewidths=0.5, ax=ax)
ax.set_title("Node Label Precision using \nNode2Vec $p=1,q=1$ Embedding as Feature")
plt.tight_layout()
plt.savefig("pricision_emb_1_1.jpg")