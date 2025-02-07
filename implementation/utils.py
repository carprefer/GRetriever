import torch
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import numpy as np

def invertDict(d):
    return {v:k for k, v in d.items()}

def getAvailableGpus():
    availableGpus = []
    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.mem_get_info(i)
        remain = stats[0]
        total = stats[1]
        if remain > total * 0.9:
            availableGpus.append(i)
    return availableGpus

def dataFrame2dictList(df):
    return [{k: df[k][i] for k in df.columns} for i in range(len(df[df.columns[0]]))]

def retrieveAndAssign(graph, qEmb, topkN=3, topkE=3, eCost=0.5):
    nodePrizes = torch.zeros(graph.num_nodes)
    edgePrizes = torch.zeros(graph.num_edges)
    c = 0.01

    cos = torch.nn.CosineSimilarity(dim=-1)

    topkN = min(topkN, graph.num_nodes)
    sims = cos(qEmb, graph.x)
    _, topkIdxs = torch.topk(sims, topkN, largest=True)
    nodePrizes[topkIdxs] = torch.arange(topkN, 0, -1).float()

    sims = cos(qEmb, graph.edge_attr)
    topkE = min(topkE, sims.unique().size(0))
    topkSims, _ = torch.topk(sims.unique(), topkE, largest=True)
    bound = topkE
    for j in range(topkE):
        topjMask = sims == topkSims[j]
        distributedPrize = min((topkE-j)/sum(topjMask), bound)
        edgePrizes[topjMask] = distributedPrize
        bound *= (1-c)
    if topkE > 0:
        eCost = min(eCost, edgePrizes.max().item()*(1-c/2))

    return nodePrizes.tolist(), edgePrizes.tolist(), eCost

def makeVirtualGraph(graph, nodePrizes, edgePrizes, eCost=0.5):
    vNodePrizes = nodePrizes
    vEdges = []
    vCosts = []
    vEid2Eid = {}
    vNid2Eid = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        if edgePrizes[i] <= eCost:
            vEid2Eid[len(vEdges)] = i
            vEdges.append((src, dst))
            vCosts.append(eCost - edgePrizes[i])
        else:
            vNodeId = len(vNodePrizes)
            vNid2Eid[vNodeId] = i
            vNodePrizes.append(edgePrizes[i] - eCost)
            vEdges.append((src, vNodeId))
            vEdges.append((vNodeId, dst))
            vCosts += [0.0, 0.0]
    
    return vNodePrizes, vEdges, vCosts, vEid2Eid, vNid2Eid

def makeSubGraph(graph, vNodePrizes, vEdges, vCosts, vEid2Eid, vNid2Eid):
    pcstNodeIdxs, pcstEdgeIdxs = pcst_fast(np.array(vEdges), np.array(vNodePrizes), np.array(vCosts), -1, 1, 'gw', 0)
    
    vNodeIdxs = pcstNodeIdxs[pcstNodeIdxs >= graph.num_nodes].tolist()
    
    subEdgeIdxs = [vEid2Eid[vEid] for vEid in pcstEdgeIdxs if vEid in vEid2Eid] + [vNid2Eid[vNid] for vNid in vNodeIdxs]
    subEdgeIdxs = list(dict.fromkeys(subEdgeIdxs))
    edgeSrcDst = graph.edge_index[:, subEdgeIdxs].tolist()
    subNodeIdxs = pcstNodeIdxs[pcstNodeIdxs < graph.num_nodes].tolist() + edgeSrcDst[0] + edgeSrcDst[1]

    subNodeIdxs = list(dict.fromkeys(subNodeIdxs))

    mapping = {nid: i for i, nid in enumerate(subNodeIdxs)}

    nodeEmbs = graph.x[subNodeIdxs]
    edgeEmbs = graph.edge_attr[subEdgeIdxs]
    edgeIdx = torch.LongTensor([[mapping[i] for i in edgeSrcDst[0]],
                                [mapping[i] for i in edgeSrcDst[1]]])

    return Data(x=nodeEmbs, edge_index=edgeIdx, edge_attr=edgeEmbs, num_nodes=len(subNodeIdxs))
