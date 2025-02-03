import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation", 
    model="meta-llama/Llama-2-7b-hf", 
    #device_map="auto",
    torch_dtype=torch.float16)
