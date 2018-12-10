import networkx as nx
import snap
import numpy as np
import pandas as pd
from sets import Set
import matplotlib.pyplot as plt

LabelName = {1: "Danaus plexippus", 2:"Heliconius charitonius", 3:"Heliconius erato", \
                4:"Junonia coenia", 5:"Lycaena phlaeas", 6:"Nymphalis antiopa", 7:"Papilio cresphontes",\
                8:"Pieris rapae", 9:"Vanessa atalanta", 10:"Vanessa cardui"}

def GetAssortivitiy(G):
    M = G.GetEdges()
    H = 0
    Sum_Pi_ki = 0
    Sum_Sum_ki_square = 0
    Sum_Sum_ki = 0
    Weighted_Sum_Pi_ki = 0
    Weighted_Sum_Sum_ki_square = 0
    Weighted_Sum_Sum_ki = 0
    for e in G.Edges():
        u = e.GetSrcNId()
        v = e.GetDstNId()
        nu = G.GetNI(u)
        nv = G.GetNI(v)
        weight = G.GetFltAttrDatE(e.GetId(), "weight")
        H += weight
        Pi_ki = nu.GetDeg() * nv.GetDeg()
        Sum_Pi_ki += Pi_ki
        Weighted_Sum_Pi_ki += weight * Pi_ki
        Sum_ki_square = (nu.GetDeg()**2) + (nv.GetDeg()**2)
        Sum_Sum_ki_square += Sum_ki_square
        Weighted_Sum_Sum_ki_square += weight * Sum_ki_square
        Sum_ki = nu.GetDeg() + nv.GetDeg()
        Sum_Sum_ki += Sum_ki
        Weighted_Sum_Sum_ki += weight * Sum_ki
    r_unweighted = (Sum_Pi_ki / float(M) - (Sum_Sum_ki / float(2*M))**2 )/ \
            (Sum_Sum_ki_square/ float(2*M) - (Sum_Sum_ki / float(2*M))**2)
    r_weighted = (Weighted_Sum_Pi_ki / float(H) - (Weighted_Sum_Sum_ki / float(2*H))**2 )/ \
            (Weighted_Sum_Sum_ki_square/ float(2*H) - (Weighted_Sum_Sum_ki / float(2*H))**2)
    rtn = (r_unweighted, r_weighted)
    print(rtn)
    return rtn

def GetNumRelatedSpecies(G):
    count = [np.zeros(11) for i in range(10)]
    for n in G.Nodes():
        s = Set()
        label = G.GetIntAttrDatN(n, "label")
        for nthId in range(n.GetDeg()):
            nid = n.GetNbrNId(nthId)
            neighbot_label = G.GetIntAttrDatN(nid, "label")
            s.add(neighbot_label)
        count[label-1][len(s)] += 1
    for dist in range(len(count)):
        count[dist] /= np.sum(count[dist])
    return count
        
def GetAndPlotNumRelatedSpecies(G):
    count = GetNumRelatedSpecies(G)
    fig = plt.figure()
    for dist in range(len(count)):
        plt.plot(count[dist], label=LabelName[dist+1])
    plt.title("# Similar Species VS Percentage of the Species")
    plt.xlabel("# similar species")
    plt.ylabel("Percentage of the Species")
    plt.legend()
    fig.savefig("RelatedSpecies.png")



def parseGraph(filename = "./GG-NE/test.tsv"):
    edgefilename =  filename # A file containing the graph, where each row contains an edge
                                     # and each edge is represented with the source and dest node ids,
                                     # the edge attributes, and the source and destination node attributes
                                     # separated by a tab.


    context = snap.TTableContext()  # When loading strings from different files, it is important to use the same context
                                    # so that SNAP knows that the same string has been seen before in another table.

    schema = snap.Schema()
    schema.Add(snap.TStrTAttrPr("srcID", snap.atInt))
    schema.Add(snap.TStrTAttrPr("dstID", snap.atInt))
    schema.Add(snap.TStrTAttrPr("weight", snap.atFlt))

    table = snap.TTable.LoadSS(schema, edgefilename, context, "\t", snap.TBool(False))
    
    # In this example, we add both edge attributes to the network,
    # but only one src node attribute, and no dst node attributes.
    edgeattrv = snap.TStrV()
    edgeattrv.Add("weight")

    srcnodeattrv = snap.TStrV()

    dstnodeattrv = snap.TStrV()

    # net will be an object of type snap.PNEANet
    G = snap.ToNetwork(snap.PNEANet, table, "srcID", "dstID", srcnodeattrv, dstnodeattrv, edgeattrv, snap.aaFirst)
    labels = pd.read_table("SS-Butterfly_labels.tsv")
    G.AddIntAttrN("label")
    for index, row in labels.iterrows():
        G.AddIntAttrDatN(row["# Node_ID"], row["Species"], "label")
    return G

#G = parseGraph()
G = parseGraph(filename = "SS-Butterfly_weights.tsv")
print ("#node", G.GetNodes())
print ("#edge", G.GetEdges())
#('#node', 25825)
#('#edge', 144593277)

GetAssortivitiy(G)
GetAndPlotNumRelatedSpecies(G)






