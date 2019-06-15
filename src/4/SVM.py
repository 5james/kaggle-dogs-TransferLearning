import logging
import tables
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from argparse import ArgumentParser
import os
import random
from sklearn.metrics import roc_auc_score, roc_curve, auc
import itertools
from sklearn.metrics.pairwise import euclidean_distances
import inspect
import sys
import thundersvm
import plotly
import plotly.graph_objs as go
import matplotlib
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA, SparsePCA
from sklearn import preprocessing

matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import interp

LOG_DIR = 'logs/'

matplotlib.use('agg')

NUM_CLASSES = 120

parser = ArgumentParser()
parser.add_argument("-e", "--experiment-dir", dest="EXPERIMENT_DIR", help="Pack whole experiment in one directory"
                                                                          "with predeclared filenames",
                    metavar="FILE", type=str, default=None)

parser.add_argument("-t", "--tsne_iterations", dest="TSNE_ITERATIONS", help="number of iterations in tsne.",
                    type=int, default=3000)
parser.add_argument("-p", "--pca_reduction", dest="PCA_REDUCTION", help="this number specify to how many dimensions"
                                                                        "pca will reduce.",
                    type=int, default=15000)
parser.add_argument("-nj", "--n_jobs", dest="N_JOBS", help="number of threads used.",
                    type=int, default=8)
parser.add_argument("-d", "--divide", dest="DIVIDE", help="On how many sets of classes should CNN codes be divided"
                                                          " (RAM problem). There are 120 classes total.",
                    type=int, default=3)
parser.add_argument("-s", "--start_from", dest="START_FROM", help="For logging purposes start from index.",
                    type=int, default=0)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    # np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.figure(figsize=(10, 10), dpi=320, facecolor='w', edgecolor='k')
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


def top_n_accuracy(preds, target, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i in range(len(target)):
        if target[i] in best_n[i, :]:
            successes += 1
    return float(successes) / target.shape[0]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


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

    # # for svm verbose
    # sl = StreamToLogger(logger, logging.INFO)
    # sys.stdout = sl
    # logger.info(args)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('-' * 90)
    logger.info(args)
    logger.info('-' * 90)

    h5_files = ['first_cnn_codes', 'second_cnn_codes']

    print('Start')

    for h5_file in h5_files:

        logger.info('Analyzing {}'.format(h5_file))

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

        # # Reducing dimensions
        # logger.info('Before:')
        # logger.info('Training CNN Codes X shape: {}'.format(X_training.shape))
        # logger.info('Training CNN Codes y shape: {}'.format(y_training.shape))
        # logger.info('Testing CNN Codes X shape: {}'.format(X_test.shape))
        # logger.info('Testing CNN Codes y shape: {}'.format(y_test.shape))
        #
        # logger.info('Start PCA')
        # pca = SparsePCA(n_components=args.PCA_REDUCTION, verbose=True, n_jobs=args.N_JOBS)
        # X_training = pca.fit_transform(X_training)
        # X_test = pca.fit_transform(X_test)
        # logger.info('Ended PCA')
        # logger.info('Start t-SNE')
        # tsne = TSNE(n_components=3, verbose=3, n_iter=args.TSNE_ITERATIONS)
        # X_training = tsne.fit_transform(X_training)
        # X_test = tsne.fit_transform(X_test)
        # logger.info('Ended t-SNE')
        #
        # logger.info('After:')
        logger.info('Training CNN Codes X shape: {}'.format(X_training.shape))
        logger.info('Training CNN Codes y shape: {}'.format(y_training.shape))
        logger.info('Testing CNN Codes X shape: {}'.format(X_test.shape))
        logger.info('Testing CNN Codes y shape: {}'.format(y_test.shape))

        mapping_idx_y_training = [[] for _ in range(120)]
        mapping_idx_y_val = [[] for _ in range(120)]

        for idx, label in enumerate(y_training):
            mapping_idx_y_training[label].append(idx)
        for idx, label in enumerate(y_test):
            mapping_idx_y_val[label].append(idx)

        testing_parameters = [{'kernel': quadraticKernel, 'degree': 2, 'coef0': 0.0},
                              {'kernel': 'linear', 'degree': 3, 'coef0': 0.0},
                              {'kernel': 'poly', 'degree': 2, 'coef0': 0.0},
                              {'kernel': 'poly', 'degree': 3, 'coef0': 0.0}]

        classes_in_subset = int(NUM_CLASSES / args.DIVIDE)

        classes_num = [str(x) for x in range(NUM_CLASSES)]

        for idx, params in enumerate(testing_parameters, start=args.START_FROM):
            logger.info('Start training SVM with params: {}'.format(params))
            avg_y_per_parameter = []
            avg_y_proba_per_parameter = []
            for subset_index in range(args.DIVIDE):
                logger.info('Training {}/{} subset'.format(subset_index + 1, args.DIVIDE))

                # Create subsets
                all_indexes_in_subset_training = np.concatenate(
                    mapping_idx_y_training[subset_index * classes_in_subset:(subset_index + 1) * classes_in_subset])

                all_indexes_in_subset_test = np.concatenate(
                    mapping_idx_y_val[subset_index * classes_in_subset:(subset_index + 1) * classes_in_subset])

                X_training_subset = np.vstack([X_training[tmp] for tmp in all_indexes_in_subset_training])
                y_training_subset = np.vstack([y_training[tmp] for tmp in all_indexes_in_subset_training]).ravel()
                X_test_subset = np.vstack([X_test[tmp] for tmp in all_indexes_in_subset_test])
                y_test_subset = np.vstack([y_test[tmp] for tmp in all_indexes_in_subset_test]).ravel()
                logger.info('Training CNN Codes X subset shape: {}'.format(X_training_subset.shape))
                logger.info('Training CNN Codes y subset shape: {}'.format(y_training_subset.shape))
                logger.info('Testing CNN Codes X subset shape: {}'.format(X_test_subset.shape))
                logger.info('Testing CNN Codes y subset shape: {}'.format(y_test_subset.shape))

                y_training_subset_proba = y_training_subset - subset_index * classes_in_subset
                y_test_subset_proba = y_test_subset - subset_index * classes_in_subset

                # Create new SVM
                svm_classifier = svm.SVC(kernel=params['kernel'],
                                         degree=params['degree'],
                                         coef0=params['coef0'],
                                         probability=True,
                                         verbose=True,
                                         cache_size=250,
                                         random_state=np.random.RandomState(0))
                # Learn SVM on subset
                svm_classifier.fit(X_training_subset, y_training_subset)
                logger.info('Ended training SVM')

                # ------------------------------------------------------------------------------------------------------
                logger.info('Start testing SVM on training data')

                y_training_prediction = svm_classifier.predict(X_training_subset)
                y_training_proba = svm_classifier.predict_proba(X_training_subset)

                # Step statistics - confusion matrix + ROC
                # confusion matrix
                confusion_matrix = metrics.confusion_matrix(y_true=y_training_subset, y_pred=y_training_prediction)
                logger.info('Confusion matrix:\n' + str(confusion_matrix))
                fig = plot_confusion_matrix(confusion_matrix,
                                            classes_num[subset_index * classes_in_subset:
                                                        (subset_index + 1) * classes_in_subset])
                plt.savefig(EXPERIMENT_DIR + h5_file + '_train-' + str(idx) + '-' + str(subset_index) + '.jpg')
                # report
                logger.info(classification_report(y_training_subset, y_training_prediction))
                # top5 accuracy

                top5 = top_n_accuracy(y_training_proba, y_training_subset_proba, 5)
                logger.info('Top5 accuray = {:.3f}'.format(top5))
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
                for current_class in range(int(NUM_CLASSES / args.DIVIDE)):
                    current_class_y_training = []
                    for jj in y_training_subset_proba:
                        current_class_y_training.append(1 if int(jj) == current_class else 0)
                    fpr, tpr, threshold = roc_curve(current_class_y_training, y_training_proba[:, current_class])
                    roc_auc = auc(fpr, tpr)
                    color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255),
                                                     random.randint(0, 255))
                    trace = go.Scatter(x=fpr, y=tpr,
                                       mode='lines',
                                       line=dict(color=color, width=1),
                                       name='{} (area = {:.3f})'.format(classes_num[current_class], float(roc_auc))
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
                                      name='average ROC curve (area = {:.3f})'.format(float(roc_auc)))
                traces.append(avg_lane)
                layout = go.Layout(title='Receiver operating characteristic',
                                   xaxis=dict(title='False Positive Rate'),
                                   yaxis=dict(title='True Positive Rate'))
                plotly.offline.plot({
                    "data": traces,
                    "layout": layout
                }, auto_open=False, filename=EXPERIMENT_DIR + '{}-{}-{}-train.html'.format(h5_file, idx, subset_index))

                # clean memory
                del y_training_prediction, y_training_proba, fpr, tpr, threshold, avg_y, avg_y_proba, traces, fig
                del confusion_matrix, layout, mid_lane

            logger.info('Ended testing new SVM on training data')

            # ------------------------------------------------------------------------------------------------------
            logger.info('Start testing new SVM on testing data')

            y_testing_prediction = svm_classifier.predict(X_test_subset)
            y_testing_proba = svm_classifier.predict_proba(X_test_subset)

            # Step statistics - confusion matrix + ROC
            # confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_true=y_test_subset, y_pred=y_testing_prediction)
            logger.info('Confusion matrix:\n' + str(confusion_matrix))
            fig = plot_confusion_matrix(confusion_matrix,
                                        classes_num[subset_index * classes_in_subset:
                                                    (subset_index + 1) * classes_in_subset])
            plt.savefig(EXPERIMENT_DIR + h5_file + '_test-' + str(idx) + '-' + str(subset_index) + '.jpg')
            # report
            logger.info(classification_report(y_test_subset, y_testing_prediction))
            # top5 accuracy
            top5 = top_n_accuracy(y_testing_proba, y_test_subset_proba, 5)
            logger.info('Top5 accuracy = {:.3f}'.format(top5))
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
            for current_class in range(int(NUM_CLASSES / args.DIVIDE)):
                current_class_y_testing = []
                for jj in y_test_subset_proba:
                    current_class_y_testing.append(1 if int(jj) == current_class else 0)
                fpr, tpr, threshold = roc_curve(current_class_y_testing, y_testing_proba[:, current_class])
                roc_auc = auc(fpr, tpr)
                color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255),
                                                 random.randint(0, 255))
                trace = go.Scatter(x=fpr, y=tpr,
                                   mode='lines',
                                   line=dict(color=color, width=1),
                                   name='{} (area = {:.3f})'.format(classes_num[current_class], float(roc_auc))
                                   )
                traces.append(trace)
                # Add to average
                avg_y.append(current_class_y_testing)
                avg_y_per_parameter.append(current_class_y_testing)
                avg_y_proba.append(y_testing_proba[:, current_class])
                avg_y_proba_per_parameter.append(y_testing_proba[:, current_class])

            avg_y = np.concatenate((avg_y), axis=0)
            avg_y_proba = np.concatenate((avg_y_proba), axis=0)

            fpr, tpr, threshold = roc_curve(avg_y, avg_y_proba)
            roc_auc = auc(fpr, tpr)
            avg_lane = go.Scatter(x=fpr, y=tpr,
                                  mode='lines',
                                  line=dict(color='deeppink', width=1, dash='dot'),
                                  name='average ROC curve (area = {:.3f})'.format(float(roc_auc)))
            traces.append(avg_lane)
            layout = go.Layout(title='Receiver operating characteristic',
                               xaxis=dict(title='False Positive Rate'),
                               yaxis=dict(title='True Positive Rate'))
            plotly.offline.plot({
                "data": traces,
                "layout": layout
            }, auto_open=False, filename=EXPERIMENT_DIR + '{}-{}-{}-test.html'.format(h5_file, idx, subset_index))

            # clean memory
            del y_testing_prediction, y_testing_proba, fpr, tpr, threshold, avg_y, avg_y_proba, traces, fig
            del confusion_matrix, layout, mid_lane

        logger.info('Ended testing new SVM on testing data')

        avg_y_per_parameter = np.concatenate((avg_y_per_parameter), axis=0)
        avg_y_proba_per_parameter = np.concatenate((avg_y_proba_per_parameter), axis=0)
        fpr, tpr, threshold = roc_curve(avg_y_per_parameter, avg_y_proba_per_parameter)
        roc_auc = auc(fpr, tpr)
        avg_lane_per_parameter = go.Scatter(x=fpr, y=tpr,
                                            mode='lines',
                                            line=dict(color='deeppink', width=1, dash='dot'),
                                            name='average ROC curve (area = {:.3f})'.format(float(roc_auc)))
        layout = go.Layout(title='Receiver operating characteristic',
                           xaxis=dict(title='False Positive Rate'),
                           yaxis=dict(title='True Positive Rate'))
        plotly.offline.plot({
            "data": [avg_lane_per_parameter],
            "layout": layout
        }, auto_open=False, filename=EXPERIMENT_DIR + '{}-{}-test.html'.format(h5_file, idx))
