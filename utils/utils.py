import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import preprocessing


def get_files(folder):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.
    Keyword arguments:
    - folder (``string``): The path to a folder.
    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    listed_files = {}

    # Explore the directory tree to get files
    for path, _, files in os.walk(folder):
        for file in files:
            if file == '.DS_Store':
                continue
            id = file.split(sep='.')[0]
            full_path = os.path.join(path, file)
            listed_files[int(id)] = full_path

    return listed_files


def read_csv(path):
    """Helper function that reads csv and returns it as numpy ndarray.
    Keyword arguments:
    - path (``string``): The path to a file.
    """
    return np.genfromtxt(path, delimiter=',')


def plot_confusion_matrix(cm):
    """Plots confusion matrix.
    Keyword arguments:
    - cm (``ndarray``): Confusion matrix.
    """
    plt.figure(1) # new numbered figure
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # plot the confusionmatrix using blue shaded colors
    plt.title("Confusion Matrix") # add title
    plt.colorbar() # plot the color bar as legend


def save_confusion_matrix(cm, epoch, mode):
    """Saves confusion matrix as a numpy file.
    Keyword arguments:
    - cm (``ndarray``): Confusion matrix.
    - epoch (``int``): Epoch associated with the confusion matrix.
    - mode (``str``): Mode (train/val/test) associated with confusion matrix.
    """
    dir = 'conf_mat'
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(f'{dir}/{mode}_{epoch}.npy', cm)


def labels_distribution(dir):
    """Counts occurring of each label and saves it to a file.
    Keyword arguments:
    - folder (``str``): Path to the directory with training data.
    """
    if not os.path.isdir(dir):
        raise RuntimeError("\"{0}\" No such directory.".format(dir))

    labels_dist = {}

    # Explore the directory tree to get files
    for path, _, files in os.walk(dir):
        for file in files:
            id = file.split(sep='.')[0]
            print(f'Reading file {id}.')
            full_path = os.path.join(path, file)
            labels = read_csv(full_path)[:, -1]

            for label in labels:
                if label in labels_dist:
                    labels_dist[label] += 1
                else:
                    labels_dist[label] = 1

    print(len(labels_dist))

    save_dir = '../stats'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    f = open(f'{save_dir}/labels_dist.pkl', "wb")
    pickle.dump(labels_dist, f)
    f.close()


def compute_weights():
    """Computes the weights for each token in vocabulary and saves them as a numpy file.
    Tokens that do not occur as labels in the data have value of x and
    the rest has values between 0 and y, computed based on frequency.
    Keyword arguments:
    - folder (``str``): Path to the directory with training data.
    """
    f = open(f'../stats/labels_dist.pkl', "rb")
    dist = pickle.load(f)
    f.close()

    vocab = np.load('../data/BigVocab_OG/voc.npy')
    ordered = sorted(dist.items(), key=lambda dict: (dict[1], dict[0]), reverse=True)
    keys = [i[0] for i in ordered]
    values = [i[1] for i in ordered]

    x = 100.0
    y = 100.0

    normalized = preprocessing.normalize(np.array(values).reshape(1, -1)).squeeze() * y
    weights = np.array([x for i in range(len(vocab) + 1)])

    for i in range(len(keys)):
        weights[int(keys[i])] = normalized[i]

    if not os.path.exists('../weights'):
        os.makedirs('../weights')

    np.save('../weights/weights.npy', weights)


def print_size_of_model(model):
    """Prints size of the model."""
    total_parameters = sum(p.numel() for p in model.parameters())
    learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total number of parameters: {total_parameters}')
    print(f'Total number of learnable parameters: {learnable_parameters}')
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


if __name__ == "__main__":
    labels_distribution('../data/BigVocab_OG/training')
