import evaluate 
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

metric = evaluate.load("sacrebleu")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

source = "hi"
target = "en"


def preprocess(file):
    en=[]
    cs=[]
    with open(file, 'r') as f:
        for l in f:
            line=l.split('\t')
            en.append(line[0])
            cs.append(line[1].strip())      
    return en, cs
en, cs = preprocess('mt_enghinglish/dev.txt')

tokenizer.src_lang = source
encoded_hi = tokenizer(cs[:10], return_tensors="pt", padding=True)
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(target))
decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print("Input: ", cs[:10])
print("Output: ", decoded_tokens)
print("Target: ", en[:10])

result = metric.compute(predictions=decoded_tokens, references=en[:10])
result = {"bleu": result["score"]}
print(result)