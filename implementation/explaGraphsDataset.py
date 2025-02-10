import pandas as pd
import re
import json
import os
from tqdm import tqdm
from gRetrieverDataset import GRetrieverDataset

RAW_PATH = "../data/explaGraphs/explaGraphs.tsv"
PATH = {
    'graphEmbs': "../data/explaGraphs/explaGraphs.pt",
    'subGraphEmbs': None,
    'qEmbs': None,
    'nodes': "../data/explaGraphs/nodes.json",
    'edges': "../data/explaGraphs/edges.json",
    'subNodes': "../data/explaGraphs/subNodes.json",
    'subEdges': "../data/explaGraphs/subEdges.json",
}

class ExplaGraphsDataset(GRetrieverDataset):
    def __init__(self, useGR=False):
        super().__init__('explaGraphs', pd.read_csv(RAW_PATH, sep='\t'), PATH, useGR=useGR)
        self.prompt = "Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:"
    
    def __getitem__(self, index):
        assert len(self.dataset) > index and len(self.graphEmbs) > index and len(self.nodes) > index and len(self.edges) > index
        
        return {
            'index': index,
            'question': f"Argument 1: {self.dataset[index]['arg1']}\n\nArgument 2: {self.dataset[index]['arg2']}\n\n{self.prompt}",
            'label': self.dataset[index]['label'],
            'graphEmbs': self.graphEmbs[index],
            'desc': self.makeDescription(index)
        }

    def extractNodesAndEdges(self):
        if os.path.exists(self.nodesPath) and os.path.exists(self.edgesPath):
            with open(self.nodesPath, 'r', encoding='utf-8') as f:
                self.nodes = json.load(f)
            with open(self.edgesPath, 'r', encoding='utf-8') as f:
                self.edges = json.load(f)
        else:
            for data in tqdm(self.dataset):
                nodes = {}
                edges = []
                for triple in re.findall(r'\((.*?)\)', data['graph']):
                    src, relation, dst = [t.lower().strip() for t in triple.split(';')]
                    if src not in nodes:
                        nodes[src] = len(nodes)
                    if dst not in nodes:
                        nodes[dst] = len(nodes)
                    edges.append({'src': nodes[src], 'edge': relation, 'dst': nodes[dst]})
                self.nodes.append(list(nodes.keys()))
                self.edges.append(edges)

            with open(self.nodesPath, 'w') as f:
                json.dump(self.nodes, f)
            with open(self.edgesPath, 'w') as f:
                json.dump(self.edges, f)
    
        self.dataId2graphId = {i: i for i in range(len(self.dataset))}

