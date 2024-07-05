#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:33:55 2024

@author: a
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
class BinaryClassifier4(nn.Module):
    '''
    Model 4
    '''
    def __init__(self, input_size, drop_val=0):
        super(BinaryClassifier4, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 32)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(32, 4)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        return x


def process_scores(scores):
    scor = []
    for score in scores:
        for s in score:
            scor.append(s[0])
    return scor

def model_scores(model, X, device):
    model.to(device)
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    scores = []
    with torch.no_grad():
        outputs = model(X_tensor)
        scores.append(outputs.cpu().numpy())
    processed_scores = process_scores(scores)
    return processed_scores

def extract_scores():
    file = "Final_outputs/LLM_representations.csv"
    print("Calculating scores")
    df = pd.DataFrame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
    model =  BinaryClassifier4(1280, 0.1386649308401776)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("Model/classifier.pth"))
    else:
        model.load_state_dict(torch.load("Model/classifier.pth", map_location=torch.device('cpu')))
    dataset = pd.read_csv(file) 
    cols = dataset.columns[3:]
    x_dataset= dataset[cols].values.astype(np.float32)
    prediction_scores = model_scores(model, x_dataset, device)
    df["Wild_type"] = dataset['Wild_type']
    df["Mutant"] = dataset['Mutant']
    df['Mutation'] = dataset['Mutation']
    df["Prediction_scores"] = prediction_scores
    df["Prediction"] = (df["Prediction_scores"] > 0.5).astype(int)
    df.to_csv("Final_outputs/Model_prediction.csv", index=False)
    print("Model Scores calculate and stored in Final Outputs: Model_prediction.csv")
