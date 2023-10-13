import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, FeatureExtractionPipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn import metrics
from sklearn.metrics import f1_score



# Load the dataset
dataset = load_dataset("lince", "sa_spaeng")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("./roberta_ft_2/checkpoint-1500", num_labels=3)

def tokenize(batch):
    return tokenizer(' '.join(batch['words']), padding='max_length')

def fix_labels(example):
    if (example['label'] == 'positive'):
        example['label'] = 1
    elif (example['label'] == 'negative'):
        example['label'] = 0
    elif (example['label'] == 'neutral'):
        example['label'] = 2
    return example


# Preprocess the dataset
tokenized_dataset = dataset.map(tokenize, batched=False)
tokenized_dataset = tokenized_dataset.rename_column("sa", "label")
tokenized_dataset = tokenized_dataset.map(fix_labels, batched=False)
tokenized_dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)


#print(tokenized_dataset["validation"])


# Inference on validation split
val_dataset = tokenized_dataset["validation"]
val_dataset = val_dataset.remove_columns(['words', 'idx', 'lid'])
print(val_dataset)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
val_preds = []
val_labels = []
##use cuda
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()
for batch in tqdm(val_dataloader):
    val_labels.extend(batch['label'].numpy().tolist())
    # Delete the label from the batch
    del batch['label']

    ##Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        print(logits)
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        print(preds)
        break
        val_preds.extend(preds)
print(val_preds, val_labels)
## Compute accuracy, precision, recall, f1
accuracy = metrics.accuracy_score(val_labels, val_preds)
precision = metrics.precision_score(val_labels, val_preds, average='macro')
recall = metrics.recall_score(val_labels, val_preds, average='macro')
f1 = metrics.f1_score(val_labels, val_preds, average='macro')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)

