import torch
import pandas as pd
import re
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data.data import Data
from datasets import load_dataset, concatenate_datasets
from textEmbedder import textEmbedder
from evaluate import eval_funcs
from utils import *
from retriever import *

DATASET_NAME = "rmanluo/RoG-webqsp"
GRAPH_EMBEDDING_PATH = "/mnt/sde/shcha/webQspGraphs.pt"
RETRIEVED_GRAPH_EMBEDDING_PATH = "../data/webQsp/retrievedWebQspGraphs.pt"
QUESTION_EMBEDDING_PATH = "../data/webQsp/questionEmbs.pt"

class WebQspDataset(Dataset):
    def __init__(self, useGR=False):
        super().__init__()
        dataset = load_dataset(DATASET_NAME)
        self.dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.prompt = "Please answer the given question."
        self.nodes, self.edges = self.extractNodesAndEdges()
        try:
            print("Loading webQsp's graph embeddings ...")
            graphPath = RETRIEVED_GRAPH_EMBEDDING_PATH if useGR else GRAPH_EMBEDDING_PATH
            self.graphEmbs = torch.load(graphPath, weights_only=False)
        except:
            print("Fail to load webQsp's graph embeddings.")
            self.graphEmbs = []
        try:
            print("Loading webQspGraphs' question embeddings ...")
            self.questionEmbs = torch.load(QUESTION_EMBEDDING_PATH, weights_only=False)
        except:
            print("Fail to load webQspGraphs' question embeddings.")
            self.questionEmbs = []

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        assert len(self.dataset) > index and len(self.graphEmbs) > index and len(self.nodes) > index and len(self.edges) > index
        nodesDf = pd.DataFrame(invertDict(self.nodes[index]).items(), columns=['node_id', 'node_attr'])
        edgesDf = pd.DataFrame(self.edges[index])
        #subg, desc = retrieval_via_pcst(self.graphEmbs[index], self.questionEmbs[index], nodesDf, edgesDf, topk=3, topk_e=3, cost_e=0.5)
        return {
            'index': index,
            'question': f"Question: {self.dataset[index]['question']}\n\nAnswer: ",
            'label': '|'.join(self.dataset[index]['answer']).lower(),
            'graphEmbs': self.graphEmbs[index],
            'qEmbs': self.questionEmbs[index],
            #'desc': desc
            #'desc': nodesDf.to_csv(index=False)+'\n'+edgesDf.to_csv(index=False)
            'desc': nodesDf.to_csv(index=False).replace('\n','\n\n')+'\n'+edgesDf.to_csv(index=False).replace('\n','\n\n')
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
        if not os.path.exists(QUESTION_EMBEDDING_PATH):
            print("WebQspGraphs preprocessing(Question) ... ")
            processedQuestions = text2embs([data['question'] for data in self.dataset])
            torch.save(processedQuestions, QUESTION_EMBEDDING_PATH)

        if not os.path.exists(GRAPH_EMBEDDING_PATH):
            print("WebQsp preprocessing(Graph) ... ")
            processedGraphs = []
            for i in tqdm(range(len(self.nodes))):
                nodeEmbs = text2embs(list(self.nodes[i].keys()))
                edgeEmbs = text2embs([e['edge'] for e in self.edges[i]])
                edgeIdx = torch.LongTensor([[e['src'] for e in self.edges[i]], 
                                            [e['dst'] for e in self.edges[i]]])
                processedGraph = Data(x=nodeEmbs, edge_index=edgeIdx, edge_attr=edgeEmbs, num_nodes=len(self.nodes[i]))
                processedGraphs.append(processedGraph)

            torch.save(processedGraphs, GRAPH_EMBEDDING_PATH)

    def splitDataset(self):
        trainTail = self.__len__() * 6 // 10
        validationTail = self.__len__() * 8 // 10
        testTail = self.__len__()
        trainIdxs = list(range(trainTail))
        validationIdxs = list(range(trainTail, validationTail))
        testIdxs = list(range(validationTail, testTail))
        return trainIdxs, validationIdxs, testIdxs

    def eval(self, path):
        return eval_funcs['webqsp'](path)