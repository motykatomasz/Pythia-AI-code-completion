import os
from shutil import copyfile
from tqdm import tqdm


def copy_to_folder(src, dest):
    files_copied = ''
    files_with_exceptions = ''
    filename = src.split("/")[-1]
    try:
        copyfile(src, dest + filename)
        files_copied = src
    except Exception:
        print(src)
        files_with_exceptions = src
        pass

    return files_copied, files_with_exceptions


def get_dataset():
    # disk = '/Volumes/Backup Plus/TUD_files/Q5/py150_files/'
    disk = '/Users/anweshcr7/Downloads/'
    dest_train = '/Users/anweshcr7/Downloads/data_train/'
    dest_eval = '/Users/anweshcr7/Downloads/data_eval/'
    train_files_copied = []
    train_files_with_exceptions = []
    eval_files_copied = []
    eval_files_with_exceptions = []

    with open(os.getcwd() + '/python100k_train.txt') as f:
        for idx, line in enumerate(tqdm(f.readlines())):
            # remove last character as its a carriage return
            files_copied, files_with_exceptions = copy_to_folder(disk + line[:-1], dest_train)
            train_files_copied.append(files_copied)
            train_files_with_exceptions.append(files_with_exceptions)

    with open(os.getcwd() + '/python50k_eval.txt') as f:
        for idx, line in enumerate(tqdm(f.readlines())):
            # remove last character as its a carriage return
            files_copied, files_with_exceptions = copy_to_folder(disk + line[:-1], dest_eval)
            eval_files_copied.append(files_copied)
            eval_files_with_exceptions.append(files_with_exceptions)

    # write the exceptions to file
    with open(os.getcwd() + "/train_files_copied.txt", "w") as outfile:
        outfile.write("\n".join(train_files_copied))

    with open(os.getcwd() + "/train_files_with_exceptions.txt", "w") as outfile:
        outfile.write("\n".join(train_files_with_exceptions))

    with open(os.getcwd() + "/eval_files_copied.txt", "w") as outfile:
        outfile.write("\n".join(eval_files_copied))

    with open(os.getcwd() + "/eval_files_with_exceptions.txt", "w") as outfile:
        outfile.write("\n".join(eval_files_with_exceptions))

    print('done copying')


if __name__ == '__main__':
    get_dataset()

