from tqdm import tqdm
from gRetrieverDataset import GRetrieverDataset
from datasets import load_dataset, concatenate_datasets

DATASET_NAME = "rmanluo/RoG-webqsp"
PATH = {
    'graphEmbs': "/mnt/sde/shcha/webQspGraphs.pt",
    'subGraphEmbs': "../data/webQsp/retrievedWebQspGraphs.pt",
    'qEmbs': "../data/webQsp/questionEmbs.pt"
}

class WebQspDataset(GRetrieverDataset):
    def __init__(self, useGR=False):
        dataset = load_dataset(DATASET_NAME)
        dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        super().__init__('webQsp', dataset, PATH, useGR=useGR, topkN=3, topkE=5, eCost=0.5)
        self.prompt = "Please answer the given question.\nAnswer:"
    
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
        for data in tqdm(self.dataset):
            nodes = {}
            edges = []
            for src, relation, dst in data['graph']:
                if src not in nodes:
                    nodes[src] = len(nodes)
                if dst not in nodes:
                    nodes[dst] = len(nodes) 
                edges.append({'src': nodes[src], 'edge': relation, 'dst': nodes[dst]})
            self.nodes.append(list(nodes.keys()))
            self.edges.append(edges)

        self.dataId2graphId = {i: i for i in range(len(self.dataset))}
