""" Custom dataset

    Comprises 20K training examples and 20K test examples
    of the CMS electromagnetic calorimeter, formatted
    as 28x28 pixel monochromatic images.

    The images are labeled as follows:

        Electron --> 0
        Photon   --> 1
        Pion     --> 2
"""
import numpy as np
import os


DATADIR = '/home/jose/work/ml-physics/data'

ELEC = 'eminus_Ele-Eta0-PhiPiOver2-Energy50.npy'
PHOT = 'gamma-Photon-Eta0-PhiPiOver2-Energy50.npy'
PION = 'piminus_Pion-Eta0-PhiPiOver2-Energy50.npy'

np.random.seed(42)


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
    """ Build dataset
    """
    X, y = read_data(threshold)

    m = min(len(y), 40000)

    X_train, X_eval = X[:m//2], X[m//2:m]
    y_train, y_eval = y[:m//2], y[m//2:m]

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
    dataset = load_dataset(threshold=10.)
    print(dataset)
    train_data = dataset.train.images
    eval_data = dataset.validation.images
    print(train_data.shape)
    print(eval_data.shape)
    print(train_data.max())
    print(eval_data.max())
