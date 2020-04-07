import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

def one_hot(y, n_classes):
    """Encode labels into ont-hot vectors
    """
    m = y.shape[0]
    y_1hot = np.zeros((m, n_classes), dtype=np.float32)
    y_1hot[np.arange(m), np.squeeze(y)] = 1
    return y_1hot

def save_plots(losses, train_accs, test_accs):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    n = len(losses)
    xs = np.arange(n)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(xs, losses, '--', linewidth=2, label='loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc='lower right')
    plt.savefig('images/loss.png')

    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_accs, '--', linewidth=2, label='train')
    ax.plot(xs, test_accs, '-', linewidth=2, label='test')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('images/accuracy.png')



def transform_labels(y, n_classes):
    y_new = np.zeros((y.shape[0], n_classes), dtype=np.int32)
    for i in range(y.shape[0]):
        y_new[i, y[i]] = 1
    return y_new


def accuracy(gold, pred):
    try:
        denom = gold.shape[0]
        nom = (gold.squeeze().long() == pred).sum()
        ret = float(nom) / denom
    except:
        denom = gold.data.shape[0]
        nom = (gold.data.squeeze().long() == pred.data).sum()
        ret = float(nom) / denom
    return ret

def bin_config(get_arg_func):
    # get arguments
    args = get_arg_func(sys.argv[1:])

    # set logger
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    formatter = logging.Formatter('[%(levelname)s][%(name)s] %(message)s')
    try:
        # if output_folder is specified in the arguments
        # put the log in there
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)
        fpath = os.path.join(args.output_folder, 'log')
    except:
        # otherwise, create a log file locally
        fpath = 'log'
    fileHandler = logging.FileHandler(fpath)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return args
