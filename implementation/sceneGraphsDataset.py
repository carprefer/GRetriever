import torch
import pandas as pd
import json
import re
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data.data import Data
from datasets import load_dataset, concatenate_datasets
from textEmbedder import textEmbedder
from evaluate import eval_funcs
from retriever import *

IMAGE_PATH = "../data/sceneGraphs/sceneGraphs.json"
QUESTION_PATH = "../data/sceneGraphs/sceneGraphsQuestions.csv"
GRAPH_EMBEDDING_PATH = "/mnt/sde/shcha/sceneGraphs.pt"
RETRIEVED_GRAPH_EMBEDDING_PATH = "../data/sceneGraphs/retrievedSceneGraphs.pt"
QUESTION_EMBEDDING_PATH = "../data/sceneGraphs/questionEmbs.pt"

class SceneGraphsDataset(Dataset):
    def __init__(self, useGR=False):
        super().__init__()
        with open(IMAGE_PATH, 'r', encoding='utf-8') as file:
            self.imageset = json.load(file)
        self.questionset = pd.read_csv(QUESTION_PATH)
        self.prompt = None
        # nodes & edges are dictionary
        self.nodes, self.edges = self.extractNodesAndEdges()
        try:
            print("Loading sceneGraphs' graph embeddings ...")
            graphPath = RETRIEVED_GRAPH_EMBEDDING_PATH if useGR else GRAPH_EMBEDDING_PATH
            self.graphEmbs = torch.load(graphPath, weights_only=False)
        except:
            print("Fail to load sceneGraphs' graph embeddings.")
            self.graphEmbs = []
        try:
            print("Loading sceneGraphs' question embeddings ...")
            self.questionEmbs = torch.load(QUESTION_EMBEDDING_PATH, weights_only=False)
        except:
            print("Fail to load sceneGraphs' question embeddings.")
            self.questionEmbs = []

    def __len__(self):
        return len(self.questionset['answer'])
    
    def __getitem__(self, index):
        assert len(self.questionset['answer']) > index
        imgIdx = self.questionset['image_id'][index]
        nodesDf = pd.DataFrame(enumerate(self.nodes[imgIdx]), columns=['node_id', 'node_attr'])
        edgesDf = pd.DataFrame(self.edges[imgIdx])
        graphEmbs = self.graphEmbs[list(self.questionset['image_id'].unique()).index(imgIdx)]
        #subg, desc = retrieval_via_pcst(graphEmbs, self.questionEmbs[index], nodesDf, edgesDf, topk=3, topk_e=3, cost_e=0.5)
        return {
            'index': index,
            'imageIdx': imgIdx,
            'question': f"Question: {self.questionset['question'][index]}\n\nAnswer: ",
            'label': self.questionset['answer'][index],
            'fullLabel': self.questionset['full_answer'][index],
            'graphEmbs': graphEmbs,
            'qEmbs': self.questionEmbs[index],
            #'desc': desc
            #'desc': nodesDf.to_csv(index=False)+'\n'+edgesDf.to_csv(index=False)
            'desc': nodesDf.to_csv(index=False).replace('\n','\n\n')+'\n'+edgesDf.to_csv(index=False).replace('\n','\n\n')
        }

    def extractNodesAndEdges(self):
        print("Extracting nodes and edges from SceneGraphs ...")
        nodesDict = {}
        edgesDict = {}    
        for imageId, image in tqdm(self.imageset.items()):
            oid2nid = {objectId:i for i, objectId in enumerate(image['objects'].keys())}
            nodes = []
            edges = []
            for objectId, object in image['objects'].items():
                nodeName = object['name']
                x, y, w, h = object['x'], object['y'], object['w'], object['h']
                nodeAttributes = object['attributes']
                nodes.append(f"name: {nodeName}; attribute: {', '.join(nodeAttributes)}; (x,y,w,h): ({x},{y},{w},{h})")

                for relation in object['relations']:
                    edges.append({'src': oid2nid[objectId], 'edge': relation['name'], 'dst': oid2nid[relation['object']]})

            nodesDict[int(imageId)] = nodes
            edgesDict[int(imageId)] = edges

        return nodesDict, edgesDict

    def preprocessing(self):
        textEmbedder.loadModel['sbert']()
        text2embs = textEmbedder.runModel['sbert']
        if not os.path.exists(QUESTION_EMBEDDING_PATH):
            print("SceneGraphs preprocessing(Question) ... ")
            processedQuestions = text2embs(self.questionset['question'].tolist())
            torch.save(processedQuestions, QUESTION_EMBEDDING_PATH)

        if not os.path.exists(GRAPH_EMBEDDING_PATH):
            print("SceneGraphs preprocessing(Graph) ... ")
            processedGraphs = []
            for imgId in tqdm(self.questionset['image_id'].unique()):
                nodeEmbs = text2embs(self.nodes[imgId])
                edgeEmbs = text2embs([e['edge'] for e in self.edges[imgId]])
                edgeIdx = torch.LongTensor([[e['src'] for e in self.edges[imgId]], 
                                            [e['dst'] for e in self.edges[imgId]]])
                processedGraph = Data(x=nodeEmbs, edge_index=edgeIdx, edge_attr=edgeEmbs, num_nodes=len(self.nodes[imgId]))
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
        return eval_funcs['scene_graphs'](path)