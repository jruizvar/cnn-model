""" Comprises 20K training examples and 20K test examples
    of the CMS electromagnetic calorimeter formatted
    as 28x28 pixel monochromatic images.

    Label convention: Electron --> 0
                      Photon   --> 1
                      Pion     --> 2
"""
from collections import namedtuple

import numpy as np
import os

np.random.seed(42)

DATADIR = '/home/jose/work/ml-physics/data'

ELEC = 'eminus_Ele-Eta0-PhiPiOver2-Energy50.npy'
PHOT = 'gamma-Photon-Eta0-PhiPiOver2-Energy50.npy'
PION = 'piminus_Pion-Eta0-PhiPiOver2-Energy50.npy'


def read_data(threshold):
    """ Read numpy arrays.
        Select images with energy above the threshold.
        Assign labels and concatenate arrays.
        Finally, shuffles and returns.
    """
    elec = np.load(os.path.join(DATADIR, ELEC))
    phot = np.load(os.path.join(DATADIR, PHOT))
    pion = np.load(os.path.join(DATADIR, PION))

    elec = np.array([i for i in elec if np.sum(i) > threshold])
    phot = np.array([i for i in phot if np.sum(i) > threshold])
    pion = np.array([i for i in pion if np.sum(i) > threshold])

    zeros = np.zeros(elec.shape[0], np.int32)
    ones = np.ones(phot.shape[0], np.int32)
    twos = 1 + np.ones(pion.shape[0], np.int32)

    y = np.concatenate((zeros, ones, twos))
    X = np.concatenate((elec, phot, pion))
    p = np.random.permutation(len(y))
    return X[p], y[p]


def load_dataset(threshold):
    """ Split into training and validation sets.
        Return namedtuple.
    """
    X, y = read_data(threshold)
    X = np.reshape(X, [-1, 28, 28, 1])
    m = min(len(y), 40000)

    X_train, X_val = X[:m//2], X[m//2:m]
    y_train, y_val = y[:m//2], y[m//2:m]

    Samples = namedtuple('Samples', 'images labels')
    Dataset = namedtuple('Dataset', 'train validation')
    return Dataset(Samples(X_train, y_train), Samples(X_val, y_val))


if __name__ == '__main__':
    dataset = load_dataset(threshold=10.)
    train_images = dataset.train.images
    print(train_images.shape)
    print(train_images.max())
