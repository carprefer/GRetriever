import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from torch_geometric.data.data import Data
from textEmbedder import textEmbedder
from evaluate import eval_funcs
from utils import *

class GRetrieverDataset(Dataset):
    def __init__(self, name, dataset, path, useGR=False):
        super().__init__()
        self.graphEmbsPath = path['retrievedGraphEmbs'] if useGR else path['graphEmbs']
        self.qEmbsPath = path['qEmbs']
        self.name = name
        self.nodes = []
        self.edges = []

        # convert dataset into list of dictionary
        if isinstance(dataset, (list, HFDataset)):
            self.dataset = dataset
        elif isinstance(dataset, pd.DataFrame):
            self.dataset = dataFrame2dictList(dataset)

        print("Extracting nodes and edges ...") 
        self.extractNodesAndEdges()
        
        self.graphEmbs = self.loadEmbeddings(self.graphEmbsPath, type='graph')
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
        return nodesDf.to_csv(index=False).replace('\n','\n\n')+'\n'+edgesDf.to_csv(index=False).replace('\n','\n\n')

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

    def splitDataset(self):
        trainTail = self.__len__() * 6 // 10
        validationTail = self.__len__() * 8 // 10
        testTail = self.__len__()
        trainIdxs = list(range(trainTail))
        validationIdxs = list(range(trainTail, validationTail))
        testIdxs = list(range(validationTail, testTail))
        return trainIdxs, validationIdxs, testIdxs
    
    def eval(self, path):
        return eval_funcs[self.name](path)

