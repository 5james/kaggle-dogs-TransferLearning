import logging
import pickle
import tables
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from argparse import ArgumentParser
import torch.nn as nn
import torch
import torchvision
from torch.autograd import Variable
import os
import random
from torch.utils.data import DataLoader, sampler, random_split, ConcatDataset
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torchvision import datasets, models, transforms
import itertools
from sklearn.metrics.pairwise import euclidean_distances
import inspect

import plotly
import plotly.graph_objs as go
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import interp

LOG_DIR = 'logs/'

matplotlib.use('agg')

NUM_CLASSES = 120


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


parser = ArgumentParser()
parser.add_argument("-e", "--experiment-dir", dest="EXPERIMENT_DIR", help="Pack whole experiment in one directory"
                                                                          "with predeclared filenames",
                    metavar="FILE", type=str, default=None)


def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    # np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.figure(figsize=(20, 20), dpi=640, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=6)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=3, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=6)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=3, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=4,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    # fig.canvas.draw()
    #
    # # Now we can save it to a numpy array.
    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return fig


def quadraticKernel(X, Y):
    C = inspect.currentframe().f_back.f_locals['self'].C
    euc = euclidean_distances(X, Y) ** 2
    return -(euc + C ** 2) ** 0.5


if __name__ == "__main__":
    args = parser.parse_args()

    EXPERIMENT_DIR = args.EXPERIMENT_DIR
    if EXPERIMENT_DIR is not None:
        if EXPERIMENT_DIR[-1] is not '/':
            EXPERIMENT_DIR += '/'
        if not os.path.isdir(EXPERIMENT_DIR):
            raise Exception("Experiment with name {} doesn't exists".format(EXPERIMENT_DIR))
    else:
        raise Exception('No experiment directory given.')

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # make sure logging directory 'logs' is available
    if not os.path.isdir(EXPERIMENT_DIR + LOG_DIR):
        os.mkdir(EXPERIMENT_DIR + LOG_DIR)

    # create file handler which logs messages
    fh = logging.FileHandler(EXPERIMENT_DIR + LOG_DIR + str(os.path.basename(__file__).split('.')[0]) + '.log')
    fh.setLevel(logging.DEBUG)

    # create console handler to print to screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('-' * 90)
    logger.info(args)
    logger.info('-' * 90)

    # check if gpu available
    use_gpu = torch.cuda.is_available() and not args.NOGPU

    h5_files = ['first_cnn_codes', 'second_cnn_codes']

    print('Start')

    for h5_file in h5_files:

        # Load train X data
        f_X_train = tables.open_file(EXPERIMENT_DIR + h5_file + '_train_X.h5', mode='r')
        X_training = f_X_train.root.data.read()
        f_X_train.close()

        # Load train y data
        f_Y_train = tables.open_file(EXPERIMENT_DIR + h5_file + '_train_y.h5', mode='r')
        y_training = f_Y_train.root.data.read().squeeze()
        f_Y_train.close()

        # Load test X data
        f_X_test_1 = tables.open_file(EXPERIMENT_DIR + h5_file + '_test_X.h5', mode='r')
        x_test_1 = f_X_test_1.root.data.read()
        f_X_test_1.close()
        f_X_test_2 = tables.open_file(EXPERIMENT_DIR + h5_file + '_val_X.h5', mode='r')
        x_test_2 = f_X_test_2.root.data.read()
        f_X_test_2.close()
        X_test = np.concatenate((x_test_1, x_test_2), axis=0)

        # Load test y data
        f_Y_test_1 = tables.open_file(EXPERIMENT_DIR + h5_file + '_test_y.h5', mode='r')
        y_test_1 = f_Y_test_1.root.data.read().squeeze()
        f_Y_test_1.close()
        f_Y_test_2 = tables.open_file(EXPERIMENT_DIR + h5_file + '_val_y.h5', mode='r')
        y_test_2 = f_Y_test_2.root.data.read().squeeze()
        f_Y_test_2.close()
        y_test = np.concatenate((y_test_1, y_test_2), axis=0)

        testing_parameters = [{'kernel': 'linear', 'degree': 3, 'coef0': 0.0},
                              {'kernel': 'poly', 'degree': 2, 'coef0': 0.0},
                              {'kernel': quadraticKernel, 'degree': 2, 'coef0': 0.0},
                              {'kernel': 'poly', 'degree': 3, 'coef0': 0.0}]

        for idx, params in enumerate(testing_parameters):
            # Create new SVM
            svm_classifier = svm.SVC(kernel=params['kernel'],
                                     degree=params['degree'],
                                     coef0=params['coef0'],
                                     probability=True,
                                     random_state=np.random.RandomState(0))
            logger.info('Start training SVM with params: {}'.format(params))
            # Learn SVM on
            svm_classifier.fit(X_training, y_training)
            logger.info('Ended training SVM')

            logger.info('Start testing SVM on training data')

            y_training_prediction = svm_classifier.predict(X_training)
            y_training_proba = svm_classifier.predict_proba(X_training)

            # Step statistics - confusion matrix + ROC
            # confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_true=y_training, y_pred=y_training_prediction)
            classes_num = [str(x) for x in range(NUM_CLASSES)]
            logger.info('Confusion matrix:\n' + str(confusion_matrix))
            fig = plot_confusion_matrix(confusion_matrix, classes_num)
            plt.savefig(EXPERIMENT_DIR + h5_file + '_train.jpg')
            # Compute ROC curve and ROC area for each class
            mid_lane = go.Scatter(x=[0, 1], y=[0, 1],
                                  mode='lines',
                                  line=dict(color='navy', width=2, dash='dash'),
                                  name='Mid-lane')
            # auc, tpr, fpr = auc_avg.value()
            # avg_lane = go.Scatter(x=fpr, y=tpr,
            #                       mode='lines',
            #                       line=dict(color='deeppink', width=1, dash='dot'),
            #                       name='average ROC curve (area = {:.2f})'.format(float(auc)))
            traces = [mid_lane]
            avg_y = []
            avg_y_proba = []
            for current_class in range(NUM_CLASSES):
                current_class_y_training = []
                for jj in y_training:
                    current_class_y_training.append(1 if int(jj) == current_class else 0)
                fpr, tpr, threshold = roc_curve(current_class_y_training, y_training_proba[:, current_class])
                roc_auc = auc(fpr, tpr)
                color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                trace = go.Scatter(x=fpr, y=tpr,
                                   mode='lines',
                                   line=dict(color=color, width=1),
                                   name='{} (area = {:.2f})'.format(classes_num[current_class], float(roc_auc))
                                   )
                traces.append(trace)
                # Add to average
                avg_y.append(current_class_y_training)
                avg_y_proba.append(y_training_proba[:, current_class])

            avg_y = np.concatenate((avg_y), axis=0)
            avg_y_proba = np.concatenate((avg_y_proba), axis=0)
            fpr, tpr, threshold = roc_curve(avg_y, avg_y_proba)
            roc_auc = auc(fpr, tpr)
            avg_lane = go.Scatter(x=fpr, y=tpr,
                                  mode='lines',
                                  line=dict(color='deeppink', width=1, dash='dot'),
                                  name='average ROC curve (area = {:.2f})'.format(float(roc_auc)))
            traces.append(avg_lane)
            layout = go.Layout(title='Receiver operating characteristic',
                               xaxis=dict(title='False Positive Rate'),
                               yaxis=dict(title='True Positive Rate'))
            plotly.offline.plot({
                "data": traces,
                "layout": layout
            }, auto_open=False, filename=EXPERIMENT_DIR + '{}-{}-train.html'.format(h5_file, idx))

            logger.info('Ended testing new SVM on training data')

            # ----------------------------------------------------------------------------------------------------------
            logger.info('Start testing new SVM on testing data')

            y_testing_prediction = svm_classifier.predict(X_test)
            y_testing_proba = svm_classifier.predict_proba(X_test)

            # Step statistics - confusion matrix + ROC
            # confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_testing_prediction)
            classes_num = [str(x) for x in range(NUM_CLASSES)]
            logger.info('Confusion matrix:\n' + str(confusion_matrix))
            fig = plot_confusion_matrix(confusion_matrix, classes_num)
            plt.savefig(EXPERIMENT_DIR + h5_file + '_test.jpg')
            # Compute ROC curve and ROC area for each class
            mid_lane = go.Scatter(x=[0, 1], y=[0, 1],
                                  mode='lines',
                                  line=dict(color='navy', width=2, dash='dash'),
                                  name='Mid-lane')
            # auc, tpr, fpr = auc_avg.value()
            # avg_lane = go.Scatter(x=fpr, y=tpr,
            #                       mode='lines',
            #                       line=dict(color='deeppink', width=1, dash='dot'),
            #                       name='average ROC curve (area = {:.2f})'.format(float(auc)))
            traces = [mid_lane]
            avg_y = []
            avg_y_proba = []
            for current_class in range(NUM_CLASSES):
                current_class_y_testing = []
                for jj in y_test:
                    current_class_y_testing.append(1 if int(jj) == current_class else 0)
                fpr, tpr, threshold = roc_curve(current_class_y_testing, y_testing_proba[:, current_class])
                roc_auc = auc(fpr, tpr)
                color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                trace = go.Scatter(x=fpr, y=tpr,
                                   mode='lines',
                                   line=dict(color=color, width=1),
                                   name='{} (area = {:.2f})'.format(classes_num[current_class], float(roc_auc))
                                   )
                traces.append(trace)
                # Add to average
                avg_y.append(current_class_y_testing)
                avg_y_proba.append(y_testing_proba[:, current_class])

            avg_y = np.concatenate((avg_y), axis=0)
            avg_y_proba = np.concatenate((avg_y_proba), axis=0)
            fpr, tpr, threshold = roc_curve(avg_y, avg_y_proba)
            roc_auc = auc(fpr, tpr)
            avg_lane = go.Scatter(x=fpr, y=tpr,
                                  mode='lines',
                                  line=dict(color='deeppink', width=1, dash='dot'),
                                  name='average ROC curve (area = {:.2f})'.format(float(roc_auc)))
            traces.append(avg_lane)
            layout = go.Layout(title='Receiver operating characteristic',
                               xaxis=dict(title='False Positive Rate'),
                               yaxis=dict(title='True Positive Rate'))
            plotly.offline.plot({
                "data": traces,
                "layout": layout
            }, auto_open=False, filename=EXPERIMENT_DIR + '{}-{}-test.html'.format(h5_file, idx))
           