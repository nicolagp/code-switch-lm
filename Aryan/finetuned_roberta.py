from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
import evaluate
import numpy as np
import os
import torch

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

train_dataset = tokenized_dataset["train"].select(range(3000))
print(train_dataset["label"])
test_dataset = tokenized_dataset["validation"]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
training_args = TrainingArguments(output_dir='roberta_ft_4/', num_train_epochs=1, report_to="none")
model.train()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model('roberta_ft_4')