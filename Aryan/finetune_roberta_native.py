from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
import evaluate
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Subset
from collections import Counter
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn import metrics
from sklearn.metrics import f1_score

os.environ['WANDB_DISABLED'] = 'true'

# Load the dataset
dataset = load_dataset("lince", "sa_spaeng")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

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
tokenized_dataset = tokenized_dataset.remove_columns(['words', 'idx', 'lid'])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# Get the UNK token ID
unk_token_id = tokenizer.unk_token_id

# Count the number of UNK tokens
num_unk_tokens = 0
total_tokens = 0
for example in tokenized_dataset['validation']:
    num_unk_tokens += torch.sum(example['input_ids'] == unk_token_id).item()
    total_tokens += torch.sum(example['input_ids'] != tokenizer.pad_token_id).item()


print("Number of UNK tokens:", num_unk_tokens)
print(num_unk_tokens / total_tokens)

train_dataset = tokenized_dataset["train"].shuffle()
#print(train_dataset["labels"])
val_test_dataset = tokenized_dataset["validation"]
test_dataset, val_dataset = val_test_dataset.train_test_split(test_size=0.3, seed=42).values()

class_counts = Counter([example['labels'].item() for example in train_dataset])
print(class_counts)
min_class_count = min(class_counts.values())
balanced_indices = []
for label in class_counts:
    print("Balancing class", label)
    # Get indices of all instances of this class
    indices = [i for i, example in enumerate(train_dataset) if example['labels'] == label]
    
    # Randomly select 'min_class_count' instances
    np.random.shuffle(indices)
    balanced_indices.extend(indices[:min_class_count])

batch_size = 8
balanced_train_dataset = Subset(train_dataset, balanced_indices)
train_dataloader = DataLoader(balanced_train_dataset, batch_size=batch_size, shuffle=True)

#train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(val_dataset, batch_size=8)
learning_rate = 1e-6
optimizer = AdamW(model.parameters(), lr=learning_rate)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model = model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
i = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        i += 1

filename = 'roberta_balanced_cs_lr_' + str(learning_rate) + 'bs_' + str(batch_size) + 'epochs_' + str(num_epochs) +'/'

model.save_pretrained(filename)

print("====Evaluating Model:", filename)

val_preds = []
val_labels = []

for batch in tqdm(eval_dataloader):
    val_labels.extend(batch['labels'].numpy().tolist())
    # Delete the label from the batch
    del batch['labels']

    ##Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        val_preds.extend(preds)
print(val_preds, val_labels)
## Compute accuracy, precision, recall, f1
accuracy = metrics.accuracy_score(val_labels, val_preds)
precision = metrics.precision_score(val_labels, val_preds, average='macro')
recall = metrics.recall_score(val_labels, val_preds, average='macro')
f1 = metrics.f1_score(val_labels, val_preds, average='macro')


print("Num samples", len(eval_dataloader))
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)