import torch
import pandas as pd
import re
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data.data import Data
from textEmbedder import textEmbedder

RAW_PATH = "../data/explaGraphs/explaGraphs.tsv"
GRAPH_PATH = "../data/explaGraphs/explaGraphs.pt"

class ExplaGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = pd.read_csv(RAW_PATH, sep='\t')
        self.prompt = "Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:"
        self.nodes, self.edges = self.extractNodesAndEdges()
        try:
            self.graphEmbs = torch.load(GRAPH_PATH, weights_only=False)
        except:
            print("Fail to load explaGraphs' graph embeddings.")
            self.graphEmbs = []

    def __len__(self):
        return len(self.dataset['label'])
    
    def __getitem__(self, index):
        assert len(self.dataset['label']) > index and len(self.graphEmbs) > index and len(self.nodes) > index and len(self.edges) > index
 
        return {
            'index': index,
            'question': f"Argument 1: {self.dataset['arg1'][index]}\nArgument 2: {self.dataset['arg2'][index]}\n{self.prompt}",
            'label': self.dataset['label'][index],
            'graphEmbs': self.graphEmbs[index],
            'nodes': self.nodes[index],
            'edges': self.edges[index]
        }

    def extractNodesAndEdges(self):
        nodesList = []
        edgesList = []    
        for graph in self.dataset['graph']:
            nodes = {}
            edges = []
            for triple in re.findall(r'\((.*?)\)', graph):
                src, relation, dst = [t.lower().strip() for t in triple.split(';')]
                if src not in nodes:
                    nodes[src] = len(nodes)
                if dst not in nodes:
                    nodes[dst] = len(nodes)
                edges.append({'src': nodes[src], 'edge': relation, 'dst': nodes[dst]})
            nodesList.append(nodes)
            edgesList.append(edges)
        return nodesList, edgesList

    def preprocessing(self):
        textEmbedder.loadModel['sbert']()
        text2embs = textEmbedder.runModel['sbert']
        print("ExplaGraphs preprocessing ... ")
        processedGraphs = []
        for i in tqdm(range(len(self.nodes))):
            nodeEmbs = text2embs(list(self.nodes[i].keys()))
            edgeEmbs = text2embs([e['edge'] for e in self.edges[i]])
            edgeIdx = torch.LongTensor([[e['src'] for e in self.edges[i]], 
                                        [e['dst'] for e in self.edges[i]]])
            processedGraph = Data(x=nodeEmbs, edge_index=edgeIdx, edge_attr=edgeEmbs, num_nodes=len(self.nodes[i]))
            processedGraphs.append(processedGraph)
        
        torch.save(processedGraphs, GRAPH_PATH)

    def splitDataset(self):
        trainTail = self.__len__() * 6 // 10
        validationTail = self.__len__() * 8 // 10
        testTail = self.__len__()
        trainIdxs = list(range(trainTail))
        validationIdxs = list(range(trainTail, validationTail))
        testIdxs = list(range(validationTail, testTail))
        return trainIdxs, validationIdxs, testIdxs


e = ExplaGraphsDataset()
e.preprocessing()
