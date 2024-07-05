#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:07:05 2024

@author: a
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir("/home/amit/Documents/PhD_work/1_Objective_first/LLM_model/Git_hub_repos")
from Data_preprocessing.Neigbhor_extraction_work import extract_neigbhours
from Data_preprocessing.ESM_LLm_representations import get_representations
from Model.model import extract_scores
import sys


directory = sys.argv[1]
file = sys.argv[2]
extract_neigbhours(directory, file)
get_representations()
extract_scores()
