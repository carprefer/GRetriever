import torch
import json
import argparse
import random
from tqdm import tqdm
from explaGraphsDataset import ExplaGraphsDataset
from sceneGraphsDataset import SceneGraphsDataset
from webQspDataset import WebQspDataset
from llm import Llm

OUTPUT_PATH = "../output/"

DATASET = {
    'expla': ExplaGraphsDataset,
    'scene': SceneGraphsDataset,
    'webQsp': WebQspDataset
}

MODEL = {
    'llm': Llm,
}

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='expla')
argparser.add_argument('--model', type=str, default='llm')
argparser.add_argument('--batchSize', type=int, default=32)
argparser.add_argument('--num', type=int, default=0)
args = argparser.parse_args()


dataset = DATASET[args.dataset]()
model = MODEL[args.model]()

print("Preprocessing ... ")
dataset.preprocessing()

print("Making test set ... ")
testIdxs = dataset.splitDataset()[2]
if args.num != 0:
    testset = [dataset[i] for i in tqdm(random.sample(testIdxs, min(len(testIdxs), args.num)))]

outputPath = OUTPUT_PATH + args.dataset + '_' + args.model + '.json'

print("Inferencing ... ")
with open(outputPath, "w") as f:
    for i in tqdm(range(0, len(testset), args.batchSize)):
        inputs = testset[i:i+args.batchSize]
        with torch.no_grad():
            outputs = model.inference(inputs)
        for input, output in zip(inputs, outputs):
            result = {
                'index': input['index'],
                'question': input['question'],
                'label': input['label'],
                'pred': output,
                'desc': input['desc']
            }
        
            f.write(json.dumps(result) + '\n')
        print(dataset.eval(outputPath))

result = dataset.eval(outputPath)
print(result)