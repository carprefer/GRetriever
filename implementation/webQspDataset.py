import torch
import pandas as pd
import re
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data.data import Data
from datasets import load_dataset, concatenate_datasets
from textEmbedder import textEmbedder

import sys

def get_size(obj, seen=None):
    """객체의 메모리 크기를 반환하는 함수"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # 객체의 ID를 seen 집합에 추가
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

DATASET_NAME = "rmanluo/RoG-webqsp"
GRAPH_EMBEDDING_PATH = "../data/webQsp/webQspGraphs.pt"
QUESTION_EMBEDDING_PATH = "../data/webQsp/questionEmbs.pt"

class WebQspDataset(Dataset):
    def __init__(self):
        super().__init__()
        dataset = load_dataset(DATASET_NAME)
        self.dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.prompt = "Please answer the given question."
        self.nodes, self.edges = self.extractNodesAndEdges()
        try:
            self.graphEmbs = torch.load(GRAPH_EMBEDDING_PATH, weights_only=False)
        except:
            print("Fail to load webQsp's graph embeddings.")
            self.graphEmbs = []
        try:
            self.questionEmbs = torch.load(QUESTION_EMBEDDING_PATH, weights_only=False)
        except:
            print("Fail to load webQspGraphs' question embeddings.")
            self.questionEmbs = []

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        assert len(self.dataset) > index and len(self.graphEmbs) > index and len(self.nodes) > index and len(self.edges) > index
 
        return {
            'index': index,
            'question': f"Question: {self.dataset[index]['question']}\nAnswer: ",
            'label': '|'.join(self.dataset[index]['answer']).lower(),
            'graphEmbs': self.graphEmbs[index],
            'qEmbs': self.questionEmbs[index],
            'nodes': self.nodes[index],
            'edges': self.edges[index]
        }

    def extractNodesAndEdges(self):
        print("Extracting nodes and edges from WebQsp ...")
        nodesList = []
        edgesList = []    
        for data in tqdm(self.dataset):
            nodes = {}
            edges = []
            for src, relation, dst in data['graph']:
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
        print("WebQspGraphs preprocessing(Question) ... ")
        processedQuestions = text2embs([data['question'] for data in self.dataset])
        torch.save(processedQuestions, QUESTION_EMBEDDING_PATH)

        print("WebQsp preprocessing(Graph) ... ")
        """processedGraphs = []
        for i in tqdm(range(len(self.nodes))):
            nodeEmbs = text2embs(list(self.nodes[i].keys()))
            edgeEmbs = text2embs([e['edge'] for e in self.edges[i]])
            edgeIdx = torch.LongTensor([[e['src'] for e in self.edges[i]], 
                                        [e['dst'] for e in self.edges[i]]])
            processedGraph = Data(x=nodeEmbs, edge_index=edgeIdx, edge_attr=edgeEmbs, num_nodes=len(self.nodes[i]))
            processedGraphs.append(processedGraph)
            print(get_size(processedGraphs))

        torch.save(processedGraphs, GRAPH_EMBEDDING_PATH)"""

    def splitDataset(self):
        trainTail = self.__len__() * 6 // 10
        validationTail = self.__len__() * 8 // 10
        testTail = self.__len__()
        trainIdxs = list(range(trainTail))
        validationIdxs = list(range(trainTail, validationTail))
        testIdxs = list(range(validationTail, testTail))
        return trainIdxs, validationIdxs, testIdxs


wq = WebQspDataset()
wq.preprocessing()