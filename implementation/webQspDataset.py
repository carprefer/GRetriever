import os
import json
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from gRetrieverDataset import GRetrieverDataset

DATASET_NAME = "rmanluo/RoG-webqsp"
PATH = {
    'graphEmbs': "/mnt/sde/shcha/webQspGraphs.pt",
    'subGraphEmbs': "../data/webQsp/retrievedWebQspGraphs.pt",
    'qEmbs': "../data/webQsp/questionEmbs.pt",
    'nodes': "../data/webQsp/nodes.json",
    'edges': "../data/webQsp/edges.json",
    'subNodes': "../data/webQsp/subNodes.json",
    'subEdges': "../data/webQsp/subEdges.json",
}

class WebQspDataset(GRetrieverDataset):
    def __init__(self, useGR=False):
        dataset = load_dataset(DATASET_NAME)
        dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        super().__init__('webQsp', dataset, PATH, useGR=useGR, topkN=3, topkE=5, eCost=0.5)
        self.prompt = "Please answer the given question."
    
    def __getitem__(self, index):
        assert len(self.dataset) > index and len(self.graphEmbs) > index and len(self.nodes) > index and len(self.edges) > index

        return {
            'index': index,
            'question': f"Question: {self.dataset[index]['question']}\n\nAnswer: ",
            'label': '|'.join(self.dataset[index]['answer']).lower(),
            'graphEmbs': self.graphEmbs[index],
            'qEmbs': self.qEmbs[index],
            'desc': self.makeDescription(index)
        }

    def extractNodesAndEdges(self):
        if self.useGR and os.path.exists(self.path['subNodes']) and os.path.exists(self.path['subEdges']):
            with open(self.path['subNodes'], 'r', encoding='utf-8') as f:
                self.nodes = json.load(f)
            with open(self.path['subEdges'], 'r', encoding='utf-8') as f:
                self.edges = json.load(f)
        elif os.path.exists(self.path['nodes']) and os.path.exists(self.path['edges']):
            with open(self.path['nodes'], 'r', encoding='utf-8') as f:
                self.nodes = json.load(f)
            with open(self.path['edges'], 'r', encoding='utf-8') as f:
                self.edges = json.load(f)
        else:
            for data in tqdm(self.dataset):
                nodes = {}
                edges = []
                for src, relation, dst in data['graph']:
                    if src not in nodes:
                        #nodes[src.lower()] = len(nodes)
                        nodes[src] = len(nodes)
                    if dst not in nodes:
                        #nodes[dst.lower()] = len(nodes)
                        nodes[dst] = len(nodes)
                    edges.append({'src': nodes[src], 'edge': relation, 'dst': nodes[dst]})
                self.nodes.append(list(nodes.keys()))
                self.edges.append(edges)
            
            with open(self.path['nodes'], 'w') as f:
                json.dump(self.nodes, f)
            with open(self.path['edges'], 'w') as f:
                json.dump(self.edges, f)

        self.dataId2graphId = {i: i for i in range(len(self.dataset))}
