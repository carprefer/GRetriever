import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gnn import GraphTransformer

class GraphLlm(torch.nn.Module):
    def __init__(self, modelName='meta-llama/Llama-2-7b-chat-hf', isFrozen=True, initPrompt=None, args=None):
        super().__init__()
        self.maxLength = args.maxLength
        self.maxNewTokens = args.maxNewTokens
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)

        self.model = AutoModelForCausalLM.from_pretrained(
            modelName,
            device_map='auto',
            #torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        if isFrozen:
            print("Freeze LLM")
            for param in self.model.parameters():
                param.requires_grad = False

        self.graphEncoder = GraphTransformer(
            in_channels=args.gnnInputDim, 
            hidden_channels=args.gnnHiddenDim, 
            out_channels=args.gnnHiddenDim, 
            num_layers=args.gnnLayerNum, 
            dropout=args.gnnDropout, 
            num_heads=args.gnnHeadNum,
        ).to(self.model.device)

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(args.gnnHiddenDim, args.gnnHiddenDim * 2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(args.gnnHiddenDim * 2, args.gnnHiddenDim * 4),
        ).to(self.model.device)

        self.embedding = self.model.model.get_input_embeddings()

        self.eosId = self.tokenizer.encode('</s>', add_special_tokens=False)
        self.bosId = self.tokenizer.encode('<s>[INST]', add_special_tokens=False)
        self.userEosId = self.tokenizer.encode('[/INST]', add_special_tokens=False)
        self.bosEmb = self.embedding(torch.tensor(self.bosId))
        self.padEmb = self.embedding(torch.tensor(0)).unsqueeze(0)
    
    def forward(self, datas=list[dict]):
        inputEmbs = []
        labelIds = []
        attentionMasks = []
        for data in datas:
            qId = self.tokenizer.encode(data['question'])
            descId = self.tokenizer.encode(data['desc'], truncation=True, max_length=self.maxLength, add_special_tokens=False)
            labelId = self.tokenizer.encode(data['label'], truncation=True, max_length=self.maxNewTokens, add_special_tokens=False)
            
            graph = data['graphEmbs'].to(self.model.device)
            graphEmb, _ = self.graphEncoder(graph.x, graph.edge_index.long(), graph.edge_attr)
            graphEmb = self.projector(graphEmb.mean(dim=0)).unsqueeze(0)

            labelId += self.eosId
            inputId = descId + qId + self.userEosId + labelId
            inputEmb = self.embedding(torch.tensor(inputId))
            inputEmb = torch.cat([self.bosEmb, graphEmb, inputEmb], dim=0)
            labelId = [-100] * (inputEmb.shape[0] - len(labelId)) + labelId

            inputEmbs.append(inputEmb)
            labelIds.append(labelId)
            attentionMasks.append([1] * inputEmb.shape[0])
        
        maxLength = max([inputEmb.shape[0] for inputEmb in inputEmbs])
        for i in range(len(inputEmbs)):
            padLength = maxLength - inputEmbs[i].shape[0]
            inputEmbs[i] = torch.cat([self.padEmb.repeat(padLength, 1), inputEmbs[i]])
            labelIds[i] = [-100] * padLength + labelIds[i]
            attentionMasks[i] = [0] * padLength + attentionMasks[i]

        outputs = self.model(
            inputs_embeds=torch.stack(inputEmbs, dim=0),
            attention_mask=torch.tensor(attentionMasks),
            labels=torch.tensor(labelIds)
        )

        return outputs.loss

    def inference(self, datas: list[dict]):
        inputEmbs = []
        attentionMasks = []
        for data in datas:
            qId = self.tokenizer.encode(data['question'])
            descId = self.tokenizer.encode(data['desc'], truncation=True, max_length=self.maxLength, add_special_tokens=False)
            
            graph = data['graphEmbs'].to(self.model.device)
            graphEmb, _ = self.graphEncoder(graph.x, graph.edge_index.long(), graph.edge_attr)
            graphEmb = self.projector(graphEmb.mean(dim=0)).unsqueeze(0)

            inputId = descId + qId + self.userEosId
            inputEmb = self.embedding(torch.tensor(inputId))
            inputEmb = torch.cat([self.bosEmb, graphEmb, inputEmb], dim=0)

            inputEmbs.append(inputEmb)
            attentionMasks.append([1] * inputEmb.shape[0])
        
        maxLength = max([inputEmb.shape[0] for inputEmb in inputEmbs])
        for i in range(len(inputEmbs)):
            padLength = maxLength - inputEmbs[i].shape[0]
            inputEmbs[i] = torch.cat([self.padEmb.repeat(padLength, 1), inputEmbs[i]])
            attentionMasks[i] = [0] * padLength + attentionMasks[i]

        outputs = self.model.generate(
            inputs_embeds=torch.stack(inputEmbs, dim=0),
            attention_mask=torch.tensor(attentionMasks),
            max_new_tokens=self.maxNewTokens,
            use_cache=True
        )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)