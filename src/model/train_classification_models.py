import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from transformer_models import SequenceClassificationDataset, SequenceClassificationDatasetNoLabels, evaluate_epoch, train_model
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from prepare_data import *
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json


def get_splits(X, y):
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    X = np.array(X)
    y = np.array(y)
    skf.get_n_splits(X, y)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        yield X_train, y_train, X_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--batch_size", default=64, type=int,
                help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--only_prediction", default=None, type=str,
                help="Epsilon for Adam optimizer.")
    parser.add_argument('--do_save', action='store_true')
    parser.add_argument("--n_labels", default="10000", type=str, help="")
    args = parser.parse_args()


    #args.save_path = "no-trainer-" + args.model_name + "-" + args.n_labels
    train, dev, test, passage2labelid, labelid2passage = load_datasets(n_labels=args.n_labels)
    
    counter = 0
    to_save = []
    out = []

    args.save_path = "finetuned-" + args.model_name + "-" + str(args.n_labels)
                
    X_train = train.destination_context.tolist()
    y_train = [passage2labelid[str(row.passage_id)] for _, row in train.iterrows()]

    X_dev = dev.destination_context.tolist()
    y_dev = [passage2labelid[str(row.passage_id)] for _, row in dev.iterrows()]
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation_side="left")
    except:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation_side="left")

    trainset = SequenceClassificationDataset(X_train, y_train, tokenizer)
    devset = SequenceClassificationDataset(X_dev, y_dev, tokenizer)
    model = train_model(trainset, args.model_name, args)

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    targets, outputs, probs = evaluate_epoch(model, devset, args)
    print ("accuracy", accuracy_score(y_dev, outputs))

    # predict dev and testset
    
    predict_dataset = SequenceClassificationDatasetNoLabels(dev.destination_context.tolist(), tokenizer)

    outputs = []
    probs = []
    hits = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(DataLoader(predict_dataset, batch_size=32, collate_fn=predict_dataset.collate_fn)):
            output = model(**batch["model_inputs"])
            argsorted = output.logits.argsort(dim=-1, descending=True).detach().cpu() # ok, argsort these
            hits_batch = []
            for i,j in zip(argsorted, output.logits):
                i = [labelid2passage[str(k.item())] for k in i[:10]]
                hits_batch.append(i[:10])
            hits.extend(hits_batch)
    true = dev.passage_id.tolist()
    with open(os.path.join(args.save_path, "predictions_devset_" + args.n_labels + ".json"), "w") as outfile:
        json.dump({"true": true, "hits": hits}, outfile)

    predict_dataset = SequenceClassificationDatasetNoLabels(test.destination_context.tolist(), tokenizer)

    outputs = []
    probs = []
    hits = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(DataLoader(predict_dataset, batch_size=32, collate_fn=predict_dataset.collate_fn)):
            output = model(**batch["model_inputs"])
            argsorted = output.logits.argsort(dim=-1, descending=True).detach().cpu() # ok, argsort these
            hits_batch = []
            for i,j in zip(argsorted, output.logits):
                i = [labelid2passage[str(k.item())] for k in i[:10]]
                hits_batch.append(i[:10])
            hits.extend(hits_batch)
    true = test.passage_id.tolist()
    with open(os.path.join(args.save_path, "predictions_testset_" + args.n_labels + ".json"), "w") as outfile:
        json.dump({"true": true, "hits": hits}, outfile)   
