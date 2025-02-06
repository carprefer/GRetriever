import pandas as pd
import json
from tqdm import tqdm
from gRetrieverDataset import GRetrieverDataset

IMAGE_PATH = "../data/sceneGraphs/sceneGraphs.json"
QUESTION_PATH = "../data/sceneGraphs/sceneGraphsQuestions.csv"
PATH = {
    'graphEmbs': "/mnt/sde/shcha/sceneGraphs.pt",
    'retrievedGraphEmbs': "../data/sceneGraphs/retrievedSceneGraphs.pt",
    'qEmbs': "../data/sceneGraphs/questionEmbs.pt"
}

class SceneGraphsDataset(GRetrieverDataset):
    def __init__(self, useGR=False):
        with open(IMAGE_PATH, 'r', encoding='utf-8') as file:
            self.imageset = json.load(file)
        self.imgId2nodesId = {}
        super().__init__('sceneGraphs', pd.read_csv(QUESTION_PATH), PATH, useGR=useGR)
        self.prompt = None

    
    def __getitem__(self, index):
        assert len(self.dataset) > index
        imgIdx = self.dataset[index]['image_id']
        nodesIdx = self.imgId2nodesId[imgIdx]
        #subg, desc = retrieval_via_pcst(graphEmbs, self.questionEmbs[index], nodesDf, edgesDf, topk=3, topk_e=3, cost_e=0.5)
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
            self.imgId2nodesId[imgId] = len(self.imgId2nodesId)

