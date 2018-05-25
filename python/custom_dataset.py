""" Custom dataset

    Comprises 45K training examples and 10K test examples
    of the CMS electromagnetic calorimeter, formatted
    as 28x28 pixel monochrome images.

    The images are labeled as follows:

        Electron --> 0
        Photon   --> 1
        Pion     --> 2
"""
from sklearn.preprocessing import binarize
import numpy as np
import os


DATADIR = '/Users/jose/Work/ml-physics/data'

ELEC = 'eminus_Ele-Eta0-PhiPiOver2-Energy50.npy'
PHOT = 'gamma-Photon-Eta0-PhiPiOver2-Energy50.npy'
PION = 'piminus_Pion-Eta0-PhiPiOver2-Energy50.npy'

np.random.seed(42)


def read_data():
    """ Read numpy arrays.
        Assign labels and concatenate arrays.
        Finally, shuffles and returns.
    """
    elec = np.load(os.path.join(DATADIR, ELEC))
    phot = np.load(os.path.join(DATADIR, PHOT))
    pion = np.load(os.path.join(DATADIR, PION))

    zeros = np.zeros(elec.shape[0], np.int32)
    ones = np.ones(phot.shape[0], np.int32)
    twos = 1 + np.ones(pion.shape[0], np.int32)

    y = np.concatenate((zeros, ones, twos))
    X = np.concatenate((elec, phot, pion))
    p = np.random.permutation(len(y))
    return X[p], y[p]


def binarizer(X, threshold):
    """ Boolean thresholding of array
    """
    return binarize(np.reshape(X, [-1, 28*28]), threshold=threshold)


def load_dataset(binarization=False, threshold=0.0001):
    """ Build dataset
    """
    X, y = read_data()

    X_train, X_eval = X[:45000], X[45000:55000]
    y_train, y_eval = y[:45000], y[45000:55000]

    if binarization:
        X_train = binarizer(X_train, threshold)
        X_eval = binarizer(X_eval, threshold)

    class Dataset(object):
        def __repr__(self):
            return self.__class__.__name__

    dataset = Dataset()
    dataset.train = Dataset()
    dataset.validation = Dataset()

    dataset.train.images = np.reshape(X_train, [-1, 28, 28, 1])
    dataset.train.labels = y_train
    dataset.train.num_examples = len(y_train)

    dataset.validation.images = np.reshape(X_eval, [-1, 28, 28, 1])
    dataset.validation.labels = y_eval
    dataset.validation.num_examples = len(y_eval)
    return dataset


if __name__ == '__main__':
    dataset = load_dataset(binarization=True)
    print(dataset)
    train_data = dataset.train.images
    train_labels = dataset.train.labels
    eval_data = dataset.validation.images
    eval_labels = dataset.validation.labels
    print(train_data.shape)
    print(eval_data.shape)
    print(train_data.max())
    print(eval_data.max())
