import community
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

LabelName = {1: "Danaus plexippus", 2:"Heliconius charitonius", 3:"Heliconius erato", \
                4:"Junonia coenia", 5:"Lycaena phlaeas", 6:"Nymphalis antiopa", 7:"Papilio cresphontes",\
                8:"Pieris rapae", 9:"Vanessa atalanta", 10:"Vanessa cardui"}


G = nx.read_edgelist("SS-Butterfly_weights.tsv", delimiter='\t', nodetype=int, data=(('weight',float),))
labels = pd.read_table("SS-Butterfly_labels.tsv")
for index, row in labels.iterrows():
    G.node[row["# Node_ID"]]["label"] = row["Species"]
    

#first compute the best partition
partition = community.best_partition(G)
colorMap = {1:'b', 2:'g', 3:'r', 4:'c', 5:'m', 6:'y', 7:'k'}
#drawing
size = float(len(set(partition.values())))
print(size)
pos = nx.spring_layout(G)
count = 0.
fig = plt.figure()
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = [count/size]*len(list_nodes), 
                                cmap=plt.cm.get_cmap('jet'), vmin=0.,vmax=1.,
                                alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.05)
plt.title("Community Detected by Louvain Algorithm")
fig.savefig("CommunityDetection.png")


ActualLabel = {1: "Danaus plexippus", 174:"Heliconius charitonius", 175:"Heliconius erato", \
                237:"Junonia coenia", 410:"Lycaena phlaeas", 513:"Nymphalis antiopa", 315:"Papilio cresphontes",\
                657:"Pieris rapae", 747:"Vanessa atalanta", 748:"Vanessa cardui"}
fig = plt.figure()
for label in range(1,11,1):
    list_nodes = labels[labels["Species"] == label]["# Node_ID"].tolist()
    p = nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20, label=LabelName[label],
                                node_color = [cm.jet(int(label/10.*255))[:3]]*len(list_nodes), 
                                #cmap=plt.cm.get_cmap('jet'), vmin=0.,vmax=1.,
                                alpha=0.8)
plt.title("Actual Labels of Butterfly Species")
nx.draw_networkx_edges(G, pos, alpha=0.05)
plt.legend()
fig.savefig("ActualCommunity.png")