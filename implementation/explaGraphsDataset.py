import pandas as pd
import re
from tqdm import tqdm
from gRetrieverDataset import GRetrieverDataset

RAW_PATH = "../data/explaGraphs/explaGraphs.tsv"
PATH = {
    'graphEmbs': "../data/explaGraphs/explaGraphs.pt",
    'retrievedGraphEmbs': None,
    'qEmbs': None
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
    

