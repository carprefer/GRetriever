import json
import random
import torch
from tqdm import tqdm
import explaGraphsDataset, sceneGraphsDataset, webQspDataset
import llm, ptLlm, graphLlm
import config, seed, utils

OUTPUT_PATH = "../output/"

DATASET = {
    'explaGraphs': explaGraphsDataset.ExplaGraphsDataset,
    'sceneGraphs': sceneGraphsDataset.SceneGraphsDataset,
    'webQsp': webQspDataset.WebQspDataset
}

MODEL = {
    'llm': llm.Llm,
    'ptLlm': ptLlm.PtLlm,
    'graphLlm': graphLlm.GraphLlm,
}

args = config.argparser.parse_args()
seed.seed_everything(seed=args.seed)

dataset = DATASET[args.dataset](useGR=args.useGR)
model = MODEL[args.model](initPrompt=dataset.prompt, args=args)

print("Preprocessing ... ")
dataset.preprocessing()

print("Making test set ... ")
testIdxs = dataset.splitDataset()[2]
if args.testNum != 0:
    testset = [dataset[i] for i in tqdm(random.sample(testIdxs, min(len(testIdxs), args.testNum)))]
else:
    testset = [dataset[i] for i in tqdm(testIdxs)]


outputPath = OUTPUT_PATH + args.dataset + '_' + args.model + ('_GR' if args.useGR else '') + '.json'

model.eval()
print("Inferencing ... ")
with open(outputPath, "w") as f:
    for i in tqdm(range(0, len(testset), args.testBatchSize)):
        inputs = testset[i:i+args.testBatchSize]
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