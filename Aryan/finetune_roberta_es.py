import xml.etree.ElementTree as ET
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import Dataset
import torch
from torch.utils.data import DataLoader, Subset
from collections import Counter
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn import metrics
from sklearn.metrics import f1_score

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length')

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    tweets = []
    labels = []

    for tweet in root.findall('tweet'):
        content = tweet.find('content').text
        sentiment_value = tweet.find('sentiment/polarity/value').text

        # Mapping sentiment to label
        if sentiment_value == 'P':
            label = 1
        elif sentiment_value == 'N':
            label = 0
        else:  # Assuming any other value is 'NONE'
            label = 2

        tweets.append(content)
        labels.append(label)

    return tweets, labels

train_tweets, train_labels = [], []
val_tweets, val_labels = [],[]

for i in range(5):
    train_filename = 'train_' + str(i) + '.xml'
    val_filename = 'dev_' + str(i) + '.xml'

    print(train_filename)
    temp_train, temp_labels = parse_xml(train_filename)
    train_tweets.extend(temp_train)
    train_labels.extend(temp_labels)

    print(val_filename)
    temp_val, val_lab = parse_xml(val_filename)
    val_tweets.extend(temp_val)
    val_labels.extend(val_lab)

test_tweets, valid_tweets, test_labels, valid_labels = train_test_split(
    val_tweets, val_labels, test_size=0.3, random_state=42
)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

# Creating datasets
train_dataset = Dataset.from_dict({'text': train_tweets, 'labels': train_labels})
test_dataset = Dataset.from_dict({'text': test_tweets, 'labels': test_labels})
val_dataset = Dataset.from_dict({'text': valid_tweets, 'labels': valid_labels})

train_dataset = train_dataset.map(tokenize, batch_size=8)
train_dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)
train_dataset = train_dataset.remove_columns(['text'])
train_dataset.set_format("torch")

val_dataset = val_dataset.map(tokenize, batch_size=8)
val_dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)
val_dataset = val_dataset.remove_columns(['text'])
val_dataset.set_format("torch")

test_dataset = test_dataset.map(tokenize, batch_size=8)
test_dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)
test_dataset = test_dataset.remove_columns(['text'])
test_dataset.set_format("torch")


print(len(train_dataset), len(test_dataset), len(val_dataset))

print(train_dataset)

batch_size = 4

# Creating DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
eval_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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

filename = 'roberta_balanced_es_lr_' + str(learning_rate) + 'bs_' + str(batch_size) + 'epochs_' + str(num_epochs) +'/'

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