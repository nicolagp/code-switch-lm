import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, FeatureExtractionPipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import f1_score

# Load the dataset
dataset = load_dataset("lince", "sa_spaeng")
##Load the multilingual XLM roberta
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")

train_dataset = dataset['train']
val_dataset = dataset['validation']
test_dataset = dataset['test']

y_train = []
y_val = []
y_test = []

X_train = []
X_val = []
X_test = []

device = 0 
#feature_extractor = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, device=device, framework='pt', return_tensors=True)

for elem in tqdm(train_dataset):
#    embed = feature_extractor(''.join(elem["words"]))
#    X_train.append(embed.mean(axis=1))
    #print(elem['sa'])
    if elem['sa'] == 'positive':
        y_train.append(1)
    elif elem['sa'] == 'neutral':
        y_train.append(0)
    elif elem['sa'] == 'negative':
        y_train.append(-1)

#print(torch.cat(X_train, dim=0).shape)

for elem in tqdm(test_dataset):
#    embed = feature_extractor(''.join(elem["words"]))
#    X_test.append(embed.mean(axis=1))
    if elem['sa'] == 'positive':
        y_test.append(1)
    elif elem['sa'] == 'neutral':
        y_test.append(0)
    elif elem['sa'] == 'negative':
        y_test.append(-1)

for elem in tqdm(val_dataset):
    #embed = feature_extractor(''.join(elem["words"]))
    #X_val.append(embed.mean(axis=1))
    if elem['sa'] == 'positive':
        y_val.append(1)
    elif elem['sa'] == 'neutral':
        y_val.append(0)
    elif elem['sa'] == 'negative':
        y_val.append(-1)

print(len(y_val), len(y_test), len(y_train))
#torch.save(torch.cat(X_train, dim=0), 'lince_sa_train.pt')
#torch.save(torch.cat(X_test, dim=0), 'lince_sa_test.pt')
#torch.save(torch.cat(X_val, dim=0), 'lince_sa_val.pt')

X_train = torch.load('lince_sa_train.pt')
X_test = torch.load('lince_sa_test.pt')
X_val = torch.load('lince_sa_val.pt')

k = 5  # Number of neighbors (you can adjust this value)
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Step 4: Evaluate the classifier's performance on the testing data
y_pred = knn_classifier.predict(X_val)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_val, y_pred)
f1_scores = f1_score(y_val, y_pred, average=None)
macro_f1 = sum(f1_scores) / len(f1_scores)
print(f"Accuracy: {accuracy}")
print("F1 Scores for each class:", f1_scores)
print("Macro F1 Score:", macro_f1)