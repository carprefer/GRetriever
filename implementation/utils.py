import torch

def invertDict(d):
    return {v:k for k, v in d.items()}

def getAvailableGpus():
    availableGpus = []
    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.mem_get_info(i)
        remain = stats[0]
        total = stats[1]
        if remain > total * 0.9:
            availableGpus.append(i)
    return availableGpus

def dataFrame2dictList(df):
    return [{k: df[k][i] for k in df.columns} for i in range(len(df[df.columns[0]]))]
