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
import torch.nn.functional as F

class Dynamic_Neural_network(nn.Module):
    def __init__(self, input_size, hidden_units, dropout_rate=0.1):
        super(Dynamic_Neural_network, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_units[0]))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_units)-1):
            self.hidden_layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output = nn.Linear(hidden_units[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for layer, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = F.leaky_relu(layer(x))
            x = dropout(x)
        x = self.sigmoid(self.output(x))
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
    hidden_layers = [45, 428, 445, 365, 84, 45]
    dropout_rate = 0.420024426418072
    model = Dynamic_Neural_network(1280, hidden_layers, dropout_rate)      
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("Model/ZFP_CanPred.pth"))
    else:
        model.load_state_dict(torch.load("Model/ZFP_CanPred.pth", map_location=torch.device('cpu')))
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
