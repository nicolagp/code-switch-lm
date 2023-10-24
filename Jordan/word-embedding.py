from transformers import AutoModel, AutoTokenizer
import numpy as np
# from utils import get_word_idx, get_word_vector,  get_hidden_states
import torch
import pandas as pd
import tqdm


model_prefix = "xlm-roberta-base"
layers = [-4, -3, -2, -1] 
model = AutoModel.from_pretrained(model_prefix)
tokenizer = AutoTokenizer.from_pretrained(model_prefix)
def get_output(encoded, model, layers):
    with torch.no_grad():
         output = model(**encoded)
    return output.pooler_output


def get_word_vector(sent, idx, tokenizer, model, layers):
    encoded = tokenizer.batch_encode_plus(sent, return_tensors="pt", padding=True)
    return get_output(encoded,model, layers)

# read data: add sentences
data = pd.read_csv("spa-eng/spa.txt", delimiter='\t', names=["eng","esp"])
eng = list(data["eng"][:1000])
esp = list(data["esp"][:1000])

cosine_list = []
# get batches of embeddings
for batch in tqdm.tqdm(range(0,len(eng),64)):
    emb_eng = get_word_vector(eng[batch:batch+64], 0, tokenizer, model, layers).cpu().detach().numpy()
    emb_esp = get_word_vector(esp[batch:batch+64], 0, tokenizer, model, layers).cpu().detach().numpy()
    # calculate cosine sim
    cosine = (emb_eng*emb_esp).sum(1)/(np.linalg.norm(emb_eng, axis=1)*np.linalg.norm(emb_esp, axis=1))
    cosine_list.append(cosine)
# print average
cosine = np.concatenate(cosine_list)
print(f'Average word similarity (cosine) metric: {np.mean(cosine)}')


