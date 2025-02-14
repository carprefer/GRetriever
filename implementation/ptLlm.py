import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class PtLlm(torch.nn.Module):
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

        self.embedding = self.model.model.get_input_embeddings()

        self.eosId = self.tokenizer.encode('</s>', add_special_tokens=False)
        self.bosId = self.tokenizer.encode('<s>[INST]', add_special_tokens=False)
        self.userEosId = self.tokenizer.encode('[/INST]', add_special_tokens=False)
        self.bosEmb = self.embedding(torch.tensor(self.bosId))
        self.padEmb = self.embedding(torch.tensor(0)).unsqueeze(0)

        # prompt tuning
        initPromptId = self.tokenizer.encode(initPrompt, add_special_tokens=False)
        initPromptId = initPromptId * args.vTokenNum
        initPromptId = initPromptId[:args.vTokenNum]

        self.initPromptEmb = torch.nn.Parameter(self.embedding.weight[torch.LongTensor(initPromptId)].detach().clone().to(torch.float32))
    
    def forward(self, datas=list[dict]):
        inputEmbs = []
        labelIds = []
        attentionMasks = []
        for data in datas:
            qId = self.tokenizer.encode(data['question'])
            descId = self.tokenizer.encode(data['desc'], truncation=True, max_length=self.maxLength, add_special_tokens=False)

            labelId = self.tokenizer.encode(data['label'], truncation=True, max_length=self.maxNewTokens, add_special_tokens=False)
            labelId += self.eosId
            inputId = descId + qId + self.userEosId + labelId
            inputEmb = self.embedding(torch.tensor(inputId))
            inputEmb = torch.cat([self.bosEmb, self.initPromptEmb, inputEmb], dim=0)
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
            
            inputId = descId + qId + self.userEosId
            inputEmb = self.embedding(torch.tensor(inputId))
            inputEmb = torch.cat([self.bosEmb, self.initPromptEmb, inputEmb], dim=0)

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
