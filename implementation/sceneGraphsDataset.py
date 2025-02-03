import torch
import pandas as pd
import json
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

IMAGE_PATH = "../data/sceneGraphs/sceneGraphs.json"
QUESTION_PATH = "../data/sceneGraphs/sceneGraphsQuestions.csv"
GRAPH_EMBEDDING_PATH = "../data/sceneGraphs/sceneGraphs.pt"
QUESTION_EMBEDDING_PATH = "../data/sceneGraphs/questionEmbs.pt"

class SceneGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open(IMAGE_PATH, 'r', encoding='utf-8') as file:
            self.imageset = json.load(file)
        self.questionset = pd.read_csv(QUESTION_PATH)
        self.prompt = None
        # nodes & edges are dictionary
        self.nodes, self.edges = self.extractNodesAndEdges()
        try:
            self.graphEmbs = torch.load(GRAPH_EMBEDDING_PATH, weights_only=False)
        except:
            print("Fail to load sceneGraphs' graph embeddings.")
            self.graphEmbs = []
        try:
            self.questionEmbs = torch.load(QUESTION_EMBEDDING_PATH, weights_only=False)
        except:
            print("Fail to load sceneGraphs' question embeddings.")
            self.questionEmbs = []

    def __len__(self):
        return len(self.questionset['label'])
    
    def __getitem__(self, index):
        assert len(self.questionset['label']) > index and len(self.graphEmbs) > index and len(self.nodes) > index and len(self.edges) > index
        imgIdx = self.questionset['image_id'][index]

        return {
            'index': index,
            'imageIdx': imgIdx,
            'question': f'Question: {self.questionset['question'][index]}\nAnswer: ',
            'label': self.questionset['answer'][index],
            'fullLabel': self.questionset['full_answer'][index],
            'graphEmbs': self.graphEmbs[index],
            'qEmbs': self.questionEmbs[index],
            'nodes': self.nodes[imgIdx],
            'edges': self.edges[imgIdx]
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

            nodesDict[imageId] = nodes
            edgesDict[imageId] = edges

        return nodesDict, edgesDict

    def preprocessing(self):
        textEmbedder.loadModel['sbert']()
        text2embs = textEmbedder.runModel['sbert']
        print("SceneGraphs preprocessing(Question) ... ")
        processedQuestions = text2embs(self.questionset['question'])
        torch.save(processedQuestions, QUESTION_EMBEDDING_PATH)

        print("SceneGraphs preprocessing(Graph) ... ")
        processedGraphs = []
        for imgId in tqdm(self.questionset['image_id'].unique()):
            nodeEmbs = text2embs(self.nodes[imgId])
            edgeEmbs = text2embs([e['edge'] for e in self.edges[imgId]])
            edgeIdx = torch.LongTensor([[e['src'] for e in self.edges[imgId]], 
                                        [e['dst'] for e in self.edges[imgId]]])
            processedGraph = Data(x=nodeEmbs, edge_index=edgeIdx, edge_attr=edgeEmbs, num_nodes=len(self.nodes[imgId]))
            processedGraphs.append(processedGraph)
            print(get_size(processedGraphs))

        torch.save(processedGraphs, GRAPH_EMBEDDING_PATH)

    def splitDataset(self):
        trainTail = self.__len__() * 6 // 10
        validationTail = self.__len__() * 8 // 10
        testTail = self.__len__()
        trainIdxs = list(range(trainTail))
        validationIdxs = list(range(trainTail, validationTail))
        testIdxs = list(range(validationTail, testTail))
        return trainIdxs, validationIdxs, testIdxs

sg = SceneGraphsDataset()
sg.preprocessing()