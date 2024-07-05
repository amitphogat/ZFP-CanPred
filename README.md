# ZFP-CanPred: Predicting the Effect of Mutations in Zinc-Finger Proteins in Cancers Using Protein Language Models
**Amit Phogat, Sowmya Ramaswamy Krishnan, Medha Pandey, M. Michael Gromiha<sup>#</sup>**

**NOTE**: This code should be used for academic purposes only. This code should not be shared or used for commercial purposes without the consent of all the authors involved.

**Description**: Python implementation of ZFP-CanPred

# Usage restrictions
The code has been provided for academic purposes only.
# Prerequisites
* pandas
* numpy
* torch
* esm
* biopython
 
# Data

* Sample test data file is provided in "Data/Structures/test_file"

* Prepare Input data file format (given in test_file)
  * Description of file format:
    
    ```
    wildtype.pdb,mutant.pdb,mutation
    ```
    (check test file for more information)

    - **wildtype.pdb**: The structure of protein without mutation
    - **mutant.pdb**: The structure of protein after introducing the mutation
    - **mutation**: example, A24G, etc

**NOTE**: Mutant structures can be generated from multiple tools such as FoldX: [FoldX Suite](https://foldxsuite.crg.eu/products#foldx_suite)


# Code Usage

Sample commands:

1. Open the terminal
2. Set the variables for the folder and file:

```bash
  folder = "Data/Structures/"
  file = "Data/Structures/test_file"
```
```
* ./run.py folder file
```
* Model scores are between 0 to 1
* Prediction cut-off, If score from model is above 0.5 (>0.5), then mutation is driver, else neutral. 
For further queries related to code usage, please write to us (work.phogat@gmail.com; gromiha@iitm.ac.in).

# Citation
Please cite this article if you use the codes in this repository for your research:
