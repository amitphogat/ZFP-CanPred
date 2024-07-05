#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:19:26 2024

@author: a
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
import esm
import numpy as np

def create_batches(data_size, batch_size):
    batches = []
    for start in range(0, data_size, batch_size):
        end = min(start + batch_size, data_size)
        batches.append([start, end])
    return batches

def get_representations():
    file_loc = "Final_outputs/Neigbhour_dataset.csv"
    
    print("Started Extracting LLM representations")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    zinc_finger_dataset = pd.read_csv(file_loc)
    result_list = create_batches(zinc_finger_dataset.shape[0], 200)
    columns = ['Wild_type_Neighbor_sequence', 'Mutant_Neighbor_sequence']
    column_names = ['rep' + str(i+1) for i in range(1280)]
    wt_df = pd.DataFrame(0, columns=column_names, index = range(zinc_finger_dataset.shape[0]))
    mt_df = pd.DataFrame(0, columns=column_names, index = range(zinc_finger_dataset.shape[0]))
    for column in columns:
        if column == 'Wild_type_Neighbor_sequence':
            print("Started_working_on_WT_neigbhor_sequence")
            for batch in result_list:
                print(batch)
                df = zinc_finger_dataset[batch[0]:batch[1]]
                data = [(row['Mutation'], row[column]) for idx, row in df.iterrows()]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33]
                sequence_representations = []
                for i, tokens_len in enumerate(batch_lens):
                    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
                wt_df.iloc[batch[0]:batch[1]] = np.array(sequence_representations)
            print("Work done on WT_neigbhor_sequence")
        if column == 'Mutant_Neighbor_sequence':
            print("Started_working_on_MT_neigbhor_sequence")
            for batch in result_list:
                print(batch)
                df = zinc_finger_dataset[batch[0]:batch[1]]
                data = [(row['Mutation'], row[column]) for idx, row in df.iterrows()]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33]
                sequence_representations = []
                for i, tokens_len in enumerate(batch_lens):
                    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
                mt_df.iloc[batch[0]:batch[1]] = np.array(sequence_representations)
            print("Work done on MT_neigbhor_sequence")
    
    
# Create a pandas DataFrame with the data and column names
    final_dataset = mt_df-wt_df
    final_dataset.insert(0,"Wild_type", zinc_finger_dataset["Wild_type"]) 
    final_dataset.insert(1,"Mutant",zinc_finger_dataset["Mutant"])
    final_dataset.insert(2,"Mutation",zinc_finger_dataset["Mutation"])
    final_dataset.to_csv("Final_outputs/LLM_representations.csv", index=False)

    print("LLM representations extracted")



    
    
 
