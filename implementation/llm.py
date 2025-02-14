import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llm(torch.nn.Module):
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
    
    def forward(self, datas=list[dict]):
        inputIds = []
        labelIds = []
        for data in datas:
            qId = self.tokenizer.encode(data['question'])
            descId = self.tokenizer.encode(data['desc'], truncation=True, max_length=self.maxLength, add_special_tokens=False)

            labelId = self.tokenizer.encode(data['label'], truncation=True, max_length=self.maxNewTokens, add_special_tokens=False)
            labelId += self.eosId
            inputId = self.bosId + descId + qId + self.userEosId + labelId
            labelId = [-100] * (len(inputId) - len(labelId)) + labelId

            inputIds.append(inputId)
            labelIds.append(labelId)
        
        maxLength = max([len(input) for input in inputIds])
        for i in range(len(inputIds)):
            inputIds[i] = [0] * (maxLength - len(inputIds[i])) + inputIds[i]
            labelIds[i] = [-100] * (maxLength - len(labelIds[i])) + labelIds[i]

        attentionMasks = [[0 if id == 0 else 1 for id in inputId] for inputId in inputIds]

        outputs = self.model(
            input_ids=torch.tensor(inputIds), 
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
            
            inputId = descId + qId
            inputEmb = self.embedding(torch.tensor(inputId))

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