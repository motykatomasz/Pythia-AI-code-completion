'''
RUN this command on terminal after installing the CD4py package
cd4py --p $FOLDERPATH --ot $TOKENIZED_OUTPUT --od py_dataset_duplicates.jsonl.gz --d 1024

where,
    $FOLDERPATH = "Path to the data folder"
    A list of duplicate files will be stored in the py_dataset_duplicates.jsonl.gz file.
    $TOKENIZED_OUTPUT = "An Empty folder to store the tokenized o/p"
'''

import os
from dpu_utils.utils.dataloading import load_jsonl_gz
import random


# Need to write this function to remove the duplicate files found in the json
def get_duplicates():
    # Selects randomly a file from each cluster of duplicate files
    # clusters_rand_files = [l.pop(random.randrange(len(l))) for l in load_jsonl_gz('py_dataset_duplicates.jsonl.gz')]
    duplicate_files = [f for l in load_jsonl_gz('../py_dataset_duplicates.jsonl.gz') for f in l]
    # Stores the duplicated files-list as a .txt file
    with open(os.getcwd() + "/duplicates_default.txt", "w") as outfile:
        outfile.write("\n".join(duplicate_files))
    print("Done")

# cd4py --p "/Users/anweshcr7/Downloads/data/" --ot "/Users/anweshcr7/Downloads/training_tokens" --od py_dataset_duplicates.jsonl.gz --d 1024

if __name__ == '__main__':
    get_duplicates()