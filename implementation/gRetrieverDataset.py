import torch
import pandas as pd
import os
import random
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from torch_geometric.data.data import Data
from textEmbedder import textEmbedder
from evaluate import eval_funcs
from utils import *

class GRetrieverDataset(Dataset):
    def __init__(self, name, dataset, path, useGR=False, topkN=3, topkE=3, eCost=1):
        super().__init__()
        self.useGR = useGR
        self.graphEmbsPath = path['graphEmbs']
        self.subGraphEmbsPath = path['subGraphEmbs']
        self.qEmbsPath = path['qEmbs']
        self.nodesPath = path['nodes']
        self.edgesPath = path['edges']
        self.subNodesPath = path['subNodes']
        self.subEdgesPath = path['subEdges']
        self.name = name
        self.nodes = []
        self.edges = []
        self.dataId2graphId = {}
        self.topkN = topkN
        self.topkE = topkE
        self.eCost = eCost

        # convert dataset into list of dictionary
        if isinstance(dataset, (list, HFDataset)):
            self.dataset = dataset
        elif isinstance(dataset, pd.DataFrame):
            self.dataset = dataFrame2dictList(dataset)

        print("Extracting nodes and edges ...") 
        self.extractNodesAndEdges()
        
        self.graphEmbs = self.loadEmbeddings(self.subGraphEmbsPath if useGR else self.graphEmbsPath, type='graph')
        if self.qEmbsPath != None:
            self.qEmbs = self.loadEmbeddings(self.qEmbsPath, type='question')

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self):
        return NotImplemented
    
    def extractNodesAndEdges(self):
        return NotImplemented
    
    def loadEmbeddings(self, path, type='graph'):
        try:
            print(f"Loading {type} embeddings ...")
            return torch.load(path, weights_only=False)
        except:
            print(f"Fail to load {type} embeddings.")
            return []
        
    def collectValues(self, key):
        return [d[key] for d in self.dataset]
    
    def makeDescription(self, index):
        nodesDf = pd.DataFrame(enumerate(self.nodes[index]), columns=['node_id', 'node_attr'])
        edgesDf = pd.DataFrame(self.edges[index])
        #return nodesDf.to_csv(index=False).replace('\n','\n\n')+'\n'+edgesDf.to_csv(index=False).replace('\n','\n\n')
        return nodesDf.to_csv(index=False)+'\n'+edgesDf.to_csv(index=False)

    def preprocessing(self):
        textEmbedder.loadModel['sbert']()
        text2embs = textEmbedder.runModel['sbert']
        if self.qEmbsPath != None and not os.path.exists(self.qEmbsPath):
            processedQuestions = text2embs([d['question'] for d in tqdm(self.dataset)])
            torch.save(processedQuestions, self.qEmbsPath)

        if not os.path.exists(self.graphEmbsPath):
            processedGraphs = []
            for nodes, edges in tqdm(zip(self.nodes, self.edges)):
                nodeEmbs = text2embs(nodes)
                edgeEmbs = text2embs([e['edge'] for e in edges])
                edgeIdx = torch.LongTensor([[e['src'] for e in edges], 
                                            [e['dst'] for e in edges]])
                processedGraph = Data(x=nodeEmbs, edge_index=edgeIdx, edge_attr=edgeEmbs, num_nodes=len(nodes))
                processedGraphs.append(processedGraph)

            torch.save(processedGraphs, self.graphEmbsPath)

        if self.subGraphEmbsPath != None and not os.path.exists(self.subGraphEmbsPath):
            processedGraphs = [None for _ in self.graphEmbs]
            processedNodes = [[] for _ in self.graphEmbs]
            processedEdges = [[] for _ in self.graphEmbs]
            for i, qEmb in tqdm(enumerate(self.qEmbs)):
                dataId = self.dataId2graphId[i]
                graph = self.graphEmbs[dataId]
                if len(self.nodes[dataId]) == 0 or len(self.edges[dataId]) == 0:
                    processedGraphs[dataId] = graph
                    processedNodes[dataId] = self.nodes[dataId]
                    processedEdges[dataId] = self.edges[dataId]
                else:
                    nodePrizes, edgePrizes, eCost = retrieveAndAssign(graph, qEmb, self.topkN, self.topkE, self.eCost)
                    vNodePrizes, vEdges, vCosts, vEid2Eid, vNid2Eid = makeVirtualGraph(graph, nodePrizes, edgePrizes, eCost)
                    processedGraph, n, e = makeSubGraph(graph, self.nodes[dataId], self.edges[dataId], vNodePrizes, vEdges, vCosts, vEid2Eid, vNid2Eid)
                    processedGraphs[dataId] = processedGraph
                    processedNodes[dataId] = n
                    processedEdges[dataId] = e

            with open(self.subNodesPath, 'w') as f:
                json.dump(processedNodes, f)
            with open(self.subEdgesPath, 'w') as f:
                json.dump(processedEdges, f)
            torch.save(processedGraphs, self.subGraphEmbsPath)

    def splitDataset(self):
        idxs = list(range(self.__len__()))
        random.shuffle(idxs)
        trainTail = self.__len__() * 6 // 10
        validationTail = self.__len__() * 8 // 10
        testTail = self.__len__()
        trainIdxs = [idxs[i] for i in range(trainTail)]
        validationIdxs = [idxs[i] for i in range(trainTail, validationTail)]
        testIdxs = [idxs[i] for i in range(validationTail, testTail)]

        # Fix bug: remove the indices of the empty graphs from the val indices
        trainIdxs = [i for i in trainIdxs if i != 2937]
        validationIdxs = [i for i in validationIdxs if i != 2937]
        testIdxs = [i for i in testIdxs if i != 2937]
        return trainIdxs, validationIdxs, testIdxs
    
    def eval(self, path):
        return eval_funcs[self.name](path)
    






