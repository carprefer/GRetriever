import os
import json
import pandas as pd
from tqdm import tqdm
from gRetrieverDataset import GRetrieverDataset

IMAGE_PATH = "../data/sceneGraphs/sceneGraphs.json"
QUESTION_PATH = "../data/sceneGraphs/sceneGraphsQuestions.csv"
PATH = {
    'graphEmbs': "/mnt/sde/shcha/sceneGraphs.pt",
    'subGraphEmbs': "../data/sceneGraphs/retrievedSceneGraphs.pt",
    'qEmbs': "../data/sceneGraphs/questionEmbs.pt",
    'nodes': "../data/sceneGraphs/nodes.json",
    'edges': "../data/sceneGraphs/edges.json",
    'subNodes': "../data/sceneGraphs/subNodes.json",
    'subEdges': "../data/sceneGraphs/subEdges.json",
}

class SceneGraphsDataset(GRetrieverDataset):
    def __init__(self, useGR=False):
        with open(IMAGE_PATH, 'r', encoding='utf-8') as file:
            self.imageset = json.load(file)
        self.imgId2graphId = {}
        super().__init__('sceneGraphs', pd.read_csv(QUESTION_PATH), PATH, useGR=useGR, topkN=3, topkE=3, eCost=1)
        self.prompt = "Please answer the given question."

    
    def __getitem__(self, index):
        assert len(self.dataset) > index
        imgIdx = self.dataset[index]['image_id']
        nodesIdx = self.imgId2graphId[imgIdx]
        return {
            'index': index,
            'imageIdx': imgIdx,
            'question': f"Question: {self.dataset[index]['question']}\n\nAnswer: ",
            'label': self.dataset[index]['answer'],
            'fullLabel': self.dataset[index]['full_answer'],
            'graphEmbs': self.graphEmbs[nodesIdx],
            'qEmbs': self.qEmbs[index],
            'desc': self.makeDescription(nodesIdx)
        }

    def extractNodesAndEdges(self):
        if self.useGR and os.path.exists(self.path['subNodes']) and os.path.exists(self.path['subEdges']):
            with open(self.path['subNodes'], 'r', encoding='utf-8') as f:
                self.nodes = json.load(f)
            with open(self.path['subEdges'], 'r', encoding='utf-8') as f:
                self.edges = json.load(f)
            for imgId in list(dict.fromkeys([d['image_id'] for d in self.dataset])):
                self.imgId2graphId[imgId] = len(self.imgId2graphId)

            self.dataId2graphId = {i: self.imgId2graphId[data['image_id']] for i, data in enumerate(self.dataset)}
        elif not self.useGR and os.path.exists(self.path['nodes']) and os.path.exists(self.path['edges']):
            with open(self.path['nodes'], 'r', encoding='utf-8') as f:
                self.nodes = json.load(f)
            with open(self.path['edges'], 'r', encoding='utf-8') as f:
                self.edges = json.load(f)
            for imgId in list(dict.fromkeys([d['image_id'] for d in self.dataset])):
                self.imgId2graphId[imgId] = len(self.imgId2graphId)
        else:
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

            for imgId in list(dict.fromkeys([d['image_id'] for d in self.dataset])):
                self.nodes.append(nodesDict[imgId])
                self.edges.append(edgesDict[imgId])
                self.imgId2graphId[imgId] = len(self.imgId2graphId)

            with open(self.path['nodes'], 'w') as f:
                json.dump(self.nodes, f)
            with open(self.path['edges'], 'w') as f:
                json.dump(self.edges, f)

        self.dataId2graphId = {i: self.imgId2graphId[data['image_id']] for i, data in enumerate(self.dataset)}