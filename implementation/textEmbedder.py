import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from accelerate import infer_auto_device_map
from utils import *

MODEL_NAME = "sentence-transformers/all-roberta-large-v1"

class TextEmbedder:
    def __init__(self):
        self.loadModel = {'sbert': self.loadSbert}
        self.runModel = {'sbert': self.runSbert}
        self.model = None
        self.tokenizer = None

    def loadSbert(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME, device_map='auto')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def runSbert(self, texts, batchSize=64):
        if len(texts) == 0:
            return torch.zeros((0, 1024))
        
        embeddings = []
        for i in range(0, len(texts), batchSize):
            batchTexts = texts[i:i+batchSize]
            tokens = self.tokenizer(batchTexts, return_tensors='pt', padding=True, truncation=True)

            with torch.no_grad():
                outputs = self.model(**tokens)
            embeddings.append(F.normalize(self.mean_pooling(outputs, tokens.attention_mask), p=2, dim=1))

        return torch.cat(embeddings, dim=0)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

textEmbedder = TextEmbedder()
#textEmbedder.loadModel['sbert']()
#print(textEmbedder.runModel['sbert']("hello").shape)
