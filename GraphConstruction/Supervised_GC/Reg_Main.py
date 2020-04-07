"""
Deep Learning @ Purdue

Author: I-Ta Lee, Bruno Ribeiro
"""


import argparse
import logging
import numpy as np
import torch
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from GraphConstruction.Custom import utils, Reg_Custom
from GraphConstruction.Custom.Reg_minibatcher import MiniBatcher
from GraphConstruction.Custom.Synthetic import load_synthetic
import sys

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training for MNIST')
    parser.add_argument('-e', '--n_epochs', type=int, default=100,
                        help='number of epochs (DEFAULT: 100)')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4,
                        help='learning rate for gradient descent (DEFAULT: 1e-4)')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='gpu id to use. -1 means cpu (DEFAULT: -1)')
    parser.add_argument('-n', '--n_training_examples', type=int, default=-1,
                        help='number of training examples used. -1 means all. (DEFAULT: -1)')

    parser.add_argument('-m', '--minibatch_size', type=int, default=-1,
                        help='minibatch_size. -1 means all. (DEFAULT: -1)')
    parser.add_argument('-p', '--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adam'], default='sgd',
                        help='stochastic gradient descent optimizer (DEFAULT: sgd)')
    parser.add_argument('-r', '--l2_lambda', type=float, default=0,
                        help='the co-efficient of L2 regularization (DEFAULT: 0)')
    parser.add_argument('-o', '--dropout_rate', type=float, default=0,
                        help='dropout rate for each layer (DEFAULT: 0)')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    parser.add_argument('-c', '--CNN', action='store_true', default=False,
                        help='run CNN messages')
    parser.add_argument('-s', '--shuffleCNN', action='store_true', default=False,
                        help='shuffle messages')
    args = parser.parse_args(argv)
    return args

def create_model():
    model= Reg_Custom.MyCNNnetwork(learning_rate=args.learning_rate, gpu_id=args.gpu_id)
    return model

def main():
    # DEBUG: fix seed
    # load data
    X_train, y_train, X_test, y_test = load_synthetic(labeled=args.n_training_examples)
    args.n_training_examples=y_train.shape[0]
    n_class=3
    X_train_all=np.concatenate((X_train,X_test))
    y_train_all=np.append(y_train,np.ones(y_test.shape[0])*(-1)).astype(int)
    y_train_confidence=np.append(np.ones(y_train.shape[0]),np.zeros(y_test.shape[0]))


    y_train_1hot = utils.one_hot(y_train, n_class)
    y_test_1hot = utils.one_hot(y_test, n_class)
    y_train_all_1hot=np.concatenate((y_train_1hot,np.zeros((y_test.shape[0], n_class), dtype=int)))

    # to torch tensor
    X_train, y_train, y_train_1hot = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(y_train_1hot)
    X_train = X_train.type(torch.FloatTensor)
    X_test, y_test, y_test_1hot = torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(y_test_1hot)
    X_test = X_test.type(torch.FloatTensor)
    X_train_all, y_train_all, y_train_all_1hot = torch.from_numpy(X_train_all), torch.from_numpy(y_train_all), torch.from_numpy(y_train_all_1hot)
    X_train_all =X_train_all.type(torch.FloatTensor)

    n_examples = y_train_all.shape[0]

    logging.info("X_train shape: {}".format(X_train.shape))
    logging.info("X_test shape: {}".format(X_test.shape))
    
    # if gpu_id is specified
    if args.gpu_id != -1:
        # move all variables to cuda
        X_train = X_train.cuda(args.gpu_id)
        y_train = y_train.cuda(args.gpu_id)
        y_train_1hot = y_train_1hot.cuda(args.gpu_id)
        X_test = X_test.cuda(args.gpu_id)
        y_test = y_test.cuda(args.gpu_id)
        y_test_1hot = y_test_1hot.cuda(args.gpu_id)
        X_train_all = X_train_all.cuda(args.gpu_id)
        y_train_all = y_train_all.cuda(args.gpu_id)
        y_train_all_1hot =y_train_all_1hot.cuda(args.gpu_id)

    # create model
    # shape = [X_train.shape[1], 300, 100, n_class]
    model = create_model()

    # start training
    losses = []
    train_accs = []
    test_accs = []

    # ======================================================================
    ## Model Training
    if args.minibatch_size > 0:

        for i_epoch in range(args.n_epochs):
            batcher = MiniBatcher(args.minibatch_size, n_examples, y_train_all, train_confidence=y_train_confidence) if args.minibatch_size > 0 \
                else MiniBatcher(n_examples, n_examples)

            logging.info("---------- EPOCH {} ----------".format(i_epoch))
            t_start = time.time()
            for train_idxs in batcher.get_one_batch():
                # numpy to torch
                if args.gpu_id != -1:
                    train_idxs = train_idxs.cuda(args.gpu_id)
                # fit to the training data
                loss = model.train_one_batch(X_train_all[train_idxs], y_train_all[train_idxs], y_train_all_1hot[train_idxs],y_train_confidence[train_idxs])
                # logging.info("loss = {}".format(loss))

            model.update_confidence(X_train_all,y_train_all,y_train_all_1hot,y_train_confidence,i_epoch)

            # monitor training and testing accuracy
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = utils.accuracy(y_train, y_train_pred)
            test_acc = utils.accuracy(y_test, y_test_pred)
            logging.info("Accuracy(train) = {}".format(train_acc))
            logging.info("Accuracy(test) = {}".format(test_acc))
            # collect results for plotting for each epoch
            training_loss = model.loss(X_train, y_train, y_train_1hot)
            regulazier_loss = model.regulazer_loss(X_train_all, y_train_all, y_train_all_1hot,y_train_confidence)

            logging.info("Training loss = {}".format(training_loss))
            logging.info("Regulazier loss = {}".format(regulazier_loss))

            logging.info("Elapse {} seconds".format(time.time() - t_start))
            losses.append(training_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print("T---> ", train_acc,test_acc,loss.item())
            model.intermediate(X_train, y_train, X_test, y_test,directory="images/transform/",epoch=i_epoch)

            # plt.clf()
            # plt.scatter(X_test.numpy()[:,0],X_test.numpy()[:,1],c=y_test_pred.numpy())
            # plt.show()
            # plt.pause(0.01)

    else:
        # here we provide the full-batch version
        for i_epoch in range(args.n_epochs):
            logging.info("---------- EPOCH {} ----------".format(i_epoch))
            t_start = time.time()

            train_idxs = np.arange(n_examples)
            np.random.shuffle(train_idxs)
            train_idxs = torch.LongTensor(train_idxs)
            # numpy to torch
            if args.gpu_id != -1:
                train_idxs = train_idxs.cuda(args.gpu_id)

            # fit to the training data
            loss = model.train_one_batch(X_train[train_idxs], y_train[train_idxs], y_train_1hot[train_idxs])

            # monitor training and testing accuracy
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = utils.accuracy(y_train, y_train_pred)
            test_acc = utils.accuracy(y_test, y_test_pred)
            logging.info("loss = {}".format(loss))
            losses.append(loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            logging.info("Accuracy(train) = {}".format(train_acc))
            logging.info("Accuracy(test) = {}".format(test_acc))
            logging.info("Elapse {} seconds".format(time.time() - t_start))

    # ======================================================================
    plt.clf()
    plt.scatter(X_test.numpy()[:,0],X_test.numpy()[:,1],c=y_test_pred.numpy())
    plt.savefig('images/test.png')

    model.intermediate(X_train,y_train,X_test,y_test)


    # plot
    utils.save_plots(losses, train_accs, test_accs)


if __name__ == '__main__':
    args = utils.bin_config(get_arguments)
    main()
