import sys
import numpy as np
from glob import glob
import os
import shutil

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('Not enough arguments: [input_folder] [train_folder] [validation_folder] [percentage=0.9]')
        sys.exit(1)

    input_folder = sys.argv[1]
    train_folder = sys.argv[2]
    validation_folder = sys.argv[3]
    percentage = 0.9 if len(sys.argv) == 4 else float(sys.argv[4])

    if not os.path.isdir(input_folder):
        print(f'{input_folder} is not a directory')
        sys.exit(1)

    files = np.array(glob(os.path.join(input_folder, '*.csv')))

    if len(files) == 0:
        print(f'no csv files in {input_folder}')
        sys.exit(1)

    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)

    if not os.path.isdir(validation_folder):
        os.mkdir(validation_folder)

    choice = np.random.choice(np.arange(len(files)), size=np.round(len(files)*percentage).astype(np.int), replace=False)

    selected = np.zeros(len(files), dtype=np.bool)
    selected[choice] = True

    for i,f in enumerate(files[selected]):
        shutil.copyfile(f, os.path.join(train_folder, f'{i}.csv'))

    for i,f in enumerate(files[~selected]):
        shutil.copyfile(f, os.path.join(validation_folder, f'{i}.csv'))
