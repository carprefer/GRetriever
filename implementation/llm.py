import torch
import contextlib
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from retriever import *
import pandas as pd
from textEmbedder import textEmbedder

class Llm(torch.nn.Module):
    def __init__(self, modelName='meta-llama/Llama-2-7b-chat-hf', isFrozen=True, maxLength=512, maxNewTokens=32):
        super().__init__()
        self.maxLength = maxLength
        self.maxNewTokens = maxNewTokens
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        #self.tokenizer.pad_token_id = 0
        #self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            modelName,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        if isFrozen:
            print("Freeze LLM")
            for param in self.model.parameters():
                param.requires_grad = False

        self.generator = pipeline(
            'text-generation', 
            model=self.model, 
            tokenizer=self.tokenizer,
            device_map='auto', 
            torch_dtype=torch.float16,
            #use_cache=True
        )

        #self.word_embedding = self.model.model.get_input_embeddings()
    
    def forward(self):
        {}
    
    def inference(self, datas: list[dict]):
        inputs = []
        for data in datas:
            question = data['question']
            # truncate description to fit in max_length
            description = self.tokenizer.decode(self.tokenizer.encode(data['desc'], truncation=True, max_length=self.maxLength, add_special_tokens=False), skip_special_tokens=True)
            input = description + '\n' + question
            #input = '<s>[INST]' + description + '\n' + question + '[/INST]'
            inputs.append(input)
        
        outputs = self.generator(inputs, max_new_tokens=self.maxNewTokens)
        return [output[0]['generated_text'].replace(input, '').strip() for input, output in zip(inputs, outputs)]
    
examples = [
    {'question': "Argument 1: Cannabis should be legal.\nArgument 2: It's not a bad thing to make marijuana more available.\nQuestion: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:", 
    'desc': "node_id,node_attr\n0,cannabis\n1,marijuana\n2,legal\n3,more available\n4,good thing\n\nsrc,edge,dst\n0,synonym of,1\n2,causes,3\n1,capable of,4\n4,desires,2"
    },
    {'question': "Argument 1: Cannabis should be legal.\n\nArgument 2: It's not a bad thing to make marijuana more available.\n\nQuestion: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:", 
    'desc': "node_id,node_attr\n\n0,cannabis\n\n1,marijuana\n\n2,legal\n\n3,more available\n\n4,good thing\n\n\nsrc,edge,dst\n\n0,synonym of,1\n\n2,causes,3\n\n1,capable of,4\n\n4,desires,2"
    },
    {"question": "Question: What animal is under the table?\n\nAnswer : ",
     "desc": "node_id,node_attr\n\n0,\"name: books; attribute: ; (x,y,w,h): (126,66,249,44)\"\n\n1,\"name: clock; attribute: white; (x,y,w,h): (1,2,115,126)\"\n\n2,\"name: dress; attribute: black; (x,y,w,h): (0,166,221,171)\"\n\n3,\"name: table; attribute: wood, long; (x,y,w,h): (0,119,374,68)\"\n\n4,\"name: eye glasses; attribute: black; (x,y,w,h): (88,201,190,85)\"\n\n5,\"name: carpet; attribute: gray, shaggy; (x,y,w,h): (1,181,360,312)\"\n\n6,\"name: cat; attribute: gray, lying; (x,y,w,h): (0,126,292,266)\"\n\n7,\"name: tag; attribute: black; (x,y,w,h): (193,295,21,24)\"\n\n\nsrc,edge,dst\n\n0,on,3\n\n0,to the right of,1\n\n1,above,6\n\n1,to the left of,0\n\n1,on,3\n\n3,above,6\n\n6,lying on,5\n\n6,wearing,2\n\n6,wearing,4\n\n6,under,3\n\n6,below,1\n\n"
    },
    {"question": "Question: What animal is under the table?\n\nAnswer : ",
     "desc": "node_id,node_attr\n0,\"name: books; attribute: ; (x,y,w,h): (126,66,249,44)\"\n1,\"name: clock; attribute: white; (x,y,w,h): (1,2,115,126)\"\n2,\"name: dress; attribute: black; (x,y,w,h): (0,166,221,171)\"\n3,\"name: table; attribute: wood, long; (x,y,w,h): (0,119,374,68)\"\n4,\"name: eye glasses; attribute: black; (x,y,w,h): (88,201,190,85)\"\n5,\"name: carpet; attribute: gray, shaggy; (x,y,w,h): (1,181,360,312)\"\n6,\"name: cat; attribute: gray, lying; (x,y,w,h): (0,126,292,266)\"\n7,\"name: tag; attribute: black; (x,y,w,h): (193,295,21,24)\"\n\nsrc,edge,dst\n0,on,3\n0,to the right of,1\n1,above,6\n1,to the left of,0\n1,on,3\n3,above,6\n6,lying on,5\n6,wearing,2\n6,wearing,4\n6,under,3\n6,below,1\n",
     "nodes": [{'node_id': 0, 'node_attr': "name: books; attribute: ; (x,y,w,h): (126,66,249,44)"},{'node_id': 1, 'node_attr': "name: clock; attribute: white; (x,y,w,h): (1,2,115,126)"},{'node_id': 2, 'node_attr': "name: dress; attribute: black; (x,y,w,h): (0,166,221,171)"},{'node_id': 3, 'node_attr': "name: table; attribute: wood, long; (x,y,w,h): (0,119,374,68)"},{'node_id': 4, 'node_attr': "name: eye glasses; attribute: black; (x,y,w,h): (88,201,190,85)"},{'node_id': 5, 'node_attr': "name: carpet; attribute: gray, shaggy; (x,y,w,h): (1,181,360,312)"},{'node_id': 6, 'node_attr': "name: cat; attribute: gray, lying; (x,y,w,h): (0,126,292,266)"},{"node_id":7, 'node_attr':"name: tag; attribute: black; (x,y,w,h): (193,295,21,24)"}],
     "edges": [{'src': 0, 'edge': 'on', 'dst': 3},{'src': 0, 'edge': 'to the right of', 'dst': 1},{'src': 1, 'edge': 'above', 'dst': 6},{'src': 1, 'edge': 'to the left of', 'dst': 0},{'src': 1, 'edge': 'on', 'dst': 3},{'src': 3, 'edge': 'above', 'dst': 6},{'src': 6, 'edge': 'lying on', 'dst': 5},{'src': 6, 'edge':'wearing', 'dst': 2},{'src': 6, 'edge':'wearing', 'dst': 4},{'src': 6, 'edge':'under', 'dst': 3},{'src': 6, 'edge':'below', 'dst': 1}]
    }
]

#llm = Llm()
#print(llm.inference(examples))
