import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='explaGraphs')
argparser.add_argument('--model', type=str, default='llm')
argparser.add_argument('--useGR', action='store_true')
argparser.add_argument('--seed', type=int, default=0)

# for training
argparser.add_argument('--lr', type=float, default=1e-5)
argparser.add_argument('--wd', type=float, default=0.05)
argparser.add_argument('--trainBatchSize', type=int, default=8)
argparser.add_argument('--num_epochs', type=int, default=10)
argparser.add_argument('--warmup_epochs', type=int, default=1)
argparser.add_argument('--lrStep', type=int, default=2)
argparser.add_argument('--patience', type=int, default=2)

# for testing
argparser.add_argument('--testBatchSize', type=int, default=32)

# for llm
argparser.add_argument('--maxLength', type=int, default=512)
argparser.add_argument('--maxNewTokens', type=int, default=32)

# for prompt tuning
argparser.add_argument('--vTokenNum', type=int, default=10)

# for gnn
argparser.add_argument("--gnnLayerNum", type=int, default=4)
argparser.add_argument("--gnnInputDim", type=int, default=1024)
argparser.add_argument("--gnnHiddenDim", type=int, default=1024)
argparser.add_argument("--gnnHeadNum", type=int, default=4)
argparser.add_argument("--gnnDropout", type=float, default=0.0)

# for fast training & inference
argparser.add_argument('--trainNum', type=int, default=0)
argparser.add_argument('--validationNum', type=int, default=0)
argparser.add_argument('--testNum', type=int, default=0)
