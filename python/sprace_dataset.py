""" Sprace Dataset

    Comprises 45K training examples and 10K test examples
    of the CMS electromagnetic calorimeter, formatted
    as 28x28 pixel monochrome images.

    The images are labeled as follows:

        Electron --> 0
        Photon   --> 1
        Pion     --> 2
"""

import numpy as np
import os


DATADIR = '/Users/jose/Work/jet-images/data'

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


def load_dataset():
    """ Build dataset
    """
    X, y = read_data()

    X_train, X_eval = X[:45000], X[45000:55000]
    y_train, y_eval = y[:45000], y[45000:55000]

    class Sprace(object):
        def __repr__(self):
            return f'{self.__class__.__name__} Dataset'

    sprace = Sprace()
    sprace.train = Sprace()
    sprace.validation = Sprace()

    sprace.train.images = X_train
    sprace.train.labels = y_train
    sprace.train.num_examples = len(y_train)

    sprace.validation.images = X_eval
    sprace.validation.labels = y_eval
    sprace.validation.num_examples = len(y_eval)
    return sprace


if __name__ == '__main__':
    sprace = load_dataset()
    print(sprace)
    train_data = sprace.train.images
    train_labels = sprace.train.labels
    eval_data = sprace.validation.images
    eval_labels = sprace.validation.labels
    print(train_data.shape, train_labels.shape)
    print(eval_data.shape, eval_labels.shape)
