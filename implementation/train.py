import torch
import json
import argparse
import random
from tqdm import tqdm
from explaGraphsDataset import ExplaGraphsDataset
from sceneGraphsDataset import SceneGraphsDataset
from webQspDataset import WebQspDataset
from llm import Llm
from ptLlm import PtLlm
from torch.nn.utils import clip_grad_norm_
from lr import adjust_learning_rate

OUTPUT_PATH = "../output/"

DATASET = {
    'explaGraphs': ExplaGraphsDataset,
    'sceneGraphs': SceneGraphsDataset,
    'webQsp': WebQspDataset
}

MODEL = {
    'llm': Llm,
    'ptLlm': PtLlm,
}

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='explaGraphs')
argparser.add_argument('--model', type=str, default='llm')

argparser.add_argument('--lr', type=float, default=1e-5)
argparser.add_argument('--wd', type=float, default=0.05)
argparser.add_argument('--trainBatchSize', type=int, default=8)
argparser.add_argument('--num_epochs', type=int, default=10)
argparser.add_argument('--warmup_epochs', type=int, default=1)
argparser.add_argument('--lrStep', type=int, default=2)
argparser.add_argument('--patience', type=int, default=2)

argparser.add_argument('--testBatchSize', type=int, default=32)

argparser.add_argument("--gnnLayerNum", type=int, default=4)
argparser.add_argument("--gnnInputDim", type=int, default=1024)
argparser.add_argument("--gnnHiddenDim", type=int, default=1024)
argparser.add_argument("--gnnHeadNum", type=int, default=4)
argparser.add_argument("--gnnDropout", type=float, default=0.0)

argparser.add_argument('--trainNum', type=int, default=0)
argparser.add_argument('--validationNum', type=int, default=0)
argparser.add_argument('--testNum', type=int, default=0)
args = argparser.parse_args()


#torch.manual_seed(42)

dataset = DATASET[args.dataset]()
model = MODEL[args.model](initPrompt=dataset.prompt)

print("Preprocessing ... ")
dataset.preprocessing()

trainIdxs = dataset.splitDataset()[0]
validationIdxs = dataset.splitDataset()[1]
testIdxs = dataset.splitDataset()[2]

print("Making train set ... ")
if args.trainNum != 0:
    trainset = [dataset[i] for i in tqdm(random.sample(trainIdxs, min(len(trainIdxs), args.trainNum)))]
else:
    trainset = [dataset[i] for i in tqdm(trainIdxs)]
print("Making validation set ... ")
if args.validationNum != 0:
    validationset = [dataset[i] for i in tqdm(random.sample(validationIdxs, min(len(validationIdxs), args.validationNum)))]
else:
    validationset = [dataset[i] for i in tqdm(validationIdxs)]
print("Making test set ... ")
if args.validationNum != 0:
    testset = [dataset[i] for i in tqdm(random.sample(testIdxs, min(len(testIdxs), args.testNum)))]
else:
    testset = [dataset[i] for i in tqdm(testIdxs)]

params = [p for _, p in model.named_parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
    betas=(0.9, 0.95)
)
print(f"trainable params: {len(params)} || all params: {len(list(model.named_parameters()))}")


outputPath = OUTPUT_PATH + args.dataset + '_' + args.model + '.json'
checkpointPath = OUTPUT_PATH + args.dataset + '_' + args.model + '_checkpoint.pth'

bestValidationLoss = float('inf')
bestEpoch = 0

print("Training ... ")
for epoch in range(args.num_epochs):
    # train
    model.train()
    epochLoss, accumLoss = 0., 0.

    print(f"Epoch: {epoch+1}/{args.num_epochs}")

    b = args.trainBatchSize
    iterNum = len(trainset) // b + (len(trainset) % b > 0)
    for i in tqdm(range(iterNum)):
        optimizer.zero_grad()
        loss = model(trainset[i*b:(i+1)*b])
        loss.backward()
        
        clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

        if (i+1) % args.lrStep == 0:
            adjust_learning_rate(optimizer.param_groups[0], args.lr, i / iterNum + epoch, args)

        optimizer.step()

        epochLoss += loss.item()
        accumLoss += loss.item()

        if (i+1) % args.lrStep == 0:
            print(f"lr: {optimizer.param_groups[0]['lr']}")
            print(f"acuum loss: {accumLoss / args.lrStep}")
            accumLoss = 0.

    print(f"Train Loss: {epochLoss / iterNum}")


    # validation
    model.eval()
    validationLoss = 0.
    b = args.trainBatchSize
    iterNum = len(validationset) // b + (len(validationset) % b > 0)
    with torch.no_grad():
        for i in tqdm(range(iterNum)):
            loss = model(validationset[i*b:(i+1)*b])
            validationLoss += loss.item()
    validationLoss /= iterNum
    print(f"Validation Loss: {validationLoss}")
    
    if validationLoss < bestValidationLoss:
        bestValidationLoss = validationLoss
        bestEpoch = epoch

        params = dict(model.named_parameters())
        stateDict = model.state_dict()
        for k in list(stateDict.keys()):
            if k in params and not params[k].requires_grad:
                del stateDict[k]
        torch.save(stateDict, checkpointPath)
        
    print(f"Best Epoch: {bestEpoch+1} (Validation Loss: {bestValidationLoss})")
    
    if epoch - bestEpoch >= args.patience:
        print(f"Quick Stop!")
        break


print("Inferencing ... ")
model.load_state_dict(torch.load(checkpointPath, map_location='cpu', weights_only=False), strict=False)
model.eval()

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