import transformers
from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer
from datasets import load_dataset_builder, load_dataset
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import tqdm
import csv


model = "meta-llama/Llama-2-7b-hf"
device = "cuda:0"

tokenizer = LlamaTokenizer.from_pretrained(model)
model = LlamaForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.float16)
builder = load_dataset_builder("lince", "pos_spaeng")
train = load_dataset("lince", "pos_spaeng", split="train")

train_encodings = []

# compute perplexity for each sentence
ppl_vals = []
n = 0
for i in tqdm.tqdm(train):
    # encode sentence
    sentence = " ".join(i["words"])
    tokens = tokenizer(sentence, return_tensors="pt")
    
    input_ids = tokens["input_ids"].to(device)  
      
    # compute nll for each word in the sentence based on context
    target_ids = input_ids.clone()
    nll = []
    for j in range(2, input_ids.shape[1]):
        target_ids[:, :j-1] = -100 # don't compute loss over context
        with torch.no_grad():
            out = model(input_ids[:, :j], labels=target_ids[:, :j])
        nll.append(out.loss)
    
    # compute perplexity
    if len(nll) > 1:
        ppl = torch.exp(torch.stack(nll).mean())
        ppl_vals.append(ppl)
    
        if n % 1000 == 0:
            print(ppl)
    
    n += 1

with open("ppl_pos_spaeng.csv", mode="w", newline='') as f:
    writer = csv.writer(f)
    for i in ppl_vals:
       writer.writerow([i.item()])    
