#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:37:25 2024

@author: a
"""


import pandas as pd
import os
from Bio.PDB import PDBParser
import numpy as np


def convert_neigbhor_sequence(lst):
    three_to_one_letter = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 
         'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 
         'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    value_lst= lst
    def custom_sort(item):
        number = int("".join(filter(str.isdigit, item)))
        return number
    sorted_lst = sorted(value_lst, key = custom_sort)
    residues = [resd[0:3] for resd in sorted_lst]
    seq_resds = [three_to_one_letter[resd] for resd in residues]
    new_list = [item for sublist in [[char, " "] for char in seq_resds] for item in sublist][:-1]
    sequence = "".join(new_list)
    return sequence


def process_raw_neigbhor(wt_dct,mt_dct, mutation):
    
    one_to_three_letter_code = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
                                    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
                                    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
                                    "Y": "TYR","V": "VAL",}
    wt_resd = one_to_three_letter_code[mutation[0]] + mutation[1:-1]
    mt_resd = one_to_three_letter_code[mutation[-1]] + mutation[1:-1]
    wt_neighbors = set().union(*(inner_dict.keys() for inner_dict in wt_dct.values()))
    mt_neighbors = set().union(*(inner_dict.keys() for inner_dict in mt_dct.values()))
    wt_neighbors = list(wt_neighbors)
    mt_neighbors  = list(mt_neighbors)
    
    wt_neighbor_seq, mt_neighbor_seq = convert_neigbhor_sequence(wt_neighbors), convert_neigbhor_sequence(mt_neighbors)
    
    return wt_resd, mt_resd, wt_neighbor_seq, mt_neighbor_seq


    
def atomic_neighbhors(structure, mutation, cut_off=8):
    print(f"Extracting neighbours {structure} and {mutation}")
    '''
    Function to calculate the distance and find the neighbors of resiude provided.
    Function takes structure, residue and cut_off as arguments
    '''
    mut_num = int(mutation[1:-1])
    target_residue = structure[0]["A"][(" ", mut_num, " ")]
    residue_coords = [(atom.get_coord(), atom.id) for atom in target_residue.get_atoms()] # getting the tuple of coords and atom
#     print(residue_coords)
    neigbhors = dict() # dictionary will store the information of atomic interaction
    for atom1 in residue_coords:
        atom1_coords = atom1[0] # coordinates
        atom1_name = atom1[1]
        neigbhors[atom1_name] = dict()# atom namemutant[]
#         print(atom1_name)
        for residue in structure.get_residues():
        #     print(residue)
            residue_num_name = ""
            for atom in residue.get_atoms():
        #         print(atom)
                other_atom_coord = atom.get_coord()
                distance = np.linalg.norm(atom1_coords - other_atom_coord)
    #             lst = []
                if distance <=cut_off:
                    if residue_num_name == "":
#                         print(atom1_name)
                        residue_num_name = residue.resname + str(residue.get_id()[1])
                        neigbhors[atom1_name][residue_num_name] = list()
                        neigbhor = (atom.get_id(), distance)
                        neigbhors[atom1_name][residue_num_name].append(neigbhor)
                    else:
                        neigbhor = (atom.get_id(), distance)
                        neigbhors[atom1_name][residue_num_name].append(neigbhor)
    print(f"Extracted neighbours {structure} and {mutation}")
    return neigbhors


def llm_dataset_preparation(directory,file):
    print("Extracting Neigbhours")
    print(file)
    df =  pd.read_csv(file, sep = ",",header=None) # reading the input file
    neigbhord_df = pd.DataFrame()
    for i in range(df.shape[0]):
        wild_type_pdb = f"{directory}/{df[0][i]}"
        mutant_type_pdb = f"{directory}/{df[1][i]}"
        mutation = df[2][i]
        wild_type_structure = PDBParser(QUIET=True).get_structure("wt", wild_type_pdb)
        mut_structure = PDBParser(QUIET=True).get_structure("mt", mutant_type_pdb)
        wt_result = atomic_neighbhors(wild_type_structure, mutation)
        mt_result = atomic_neighbhors(mut_structure, mutation)
        wt_resd, mt_resd, wt_neighbor_seq, mt_neighbor_seq = process_raw_neigbhor(wt_result,mt_result, mutation)
        
        dct_df = pd.DataFrame({"Wild_type":[df[0][i]],"Mutant":[df[1][i]], "Mutation": [mutation],
                               "WT_resd":[wt_resd],"MT_resd":[mt_resd],
                               "Wild_type_Neighbor_sequence":[wt_neighbor_seq], "Mutant_Neighbor_sequence":[mt_neighbor_seq]})
        neigbhord_df = pd.concat([neigbhord_df, dct_df], ignore_index=True)

    neigbhord_df.to_csv("Final_outputs/Neigbhour_dataset.csv", index=False)
    print("Dataset saved: Neigbhour_dataset.csv")


def make_folder():
    file_name = "Final_outputs"
    # Check if the folder doesn't exist, then create it
    if not os.path.exists(file_name):
        os.makedirs(file_name)
        print(f"Folder '{file_name}' created successfully.")
    else:
        pass
    return file_name

def extract_neigbhours(directory, file):
    make_folder()
    llm_dataset_preparation(directory,file)


