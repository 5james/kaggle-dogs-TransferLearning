import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import tables
from argparse import ArgumentParser

LOG_DIR = 'logs/'

matplotlib.use('agg')

NUM_CLASSES = 120

parser = ArgumentParser()

parser.add_argument("-e", "--experiment-dir", dest="EXPERIMENT_DIR", help="Pack whole experiment in one directory",
                    metavar="FILE", type=str, default=None)
parser.add_argument("-t", "--tsne_iterations", dest="TSNE_ITERATIONS", help="number of iterations in tsne.",
                    type=int, default=1500)
parser.add_argument("-p", "--pca_reduction", dest="PCA_REDUCTION", help="this number specify to how many dimensions"
                                                                        "pca will reduce.",
                    type=int, default=150)

args = parser.parse_args()


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


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


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
    logger.info(args)

    # for tsne print -> logger
    sl = StreamToLogger(logger, logging.INFO)
    sys.stdout = sl
    logger.info(args)

    # # Find all h5 files
    # h5_files = []
    # for file in os.listdir(EXPERIMENT_DIR):
    #     if file.endswith(".h5"):
    #         h5_files.append(EXPERIMENT_DIR + file)
    #
    # # Find base of files
    # h5_files_base = [x.replace('_X.h5', '').replace('_x.h5', '').replace('_Y.h5', '').replace('_y.h5', '')
    #                  for x in h5_files]
    # # Remove all duplicates
    # h5_files_base = list(dict.fromkeys(h5_files_base))

    h5_files = [['first_cnn_codes_train'],
                ['first_cnn_codes_test', 'first_cnn_codes_val'],
                ['second_cnn_codes_train'],
                ['second_cnn_codes_test', 'second_cnn_codes_val']]

    print('Start')

    for h5_file in h5_files:

        X_training_list = []
        y_training_list = []

        for filename in h5_file:
            # Load files
            f_X = tables.open_file(EXPERIMENT_DIR + filename + '_X.h5', mode='r')
            X_training_list.append(f_X.root.data.read())
            f_X.close()

            f_Y = tables.open_file(EXPERIMENT_DIR + filename + '_y.h5', mode='r')
            y_training_list.append(f_Y.root.data.read().squeeze())
            f_Y.close()

        X_training = np.concatenate((X_training_list), axis=0)
        y_training = np.concatenate((y_training_list), axis=0)

        # PCA algorithm at first to reduce number of dimensions.
        # https://github.com/rnoxy/cifar10-cnn/blob/master/Feature_extraction_with_fine_tuned_pretrained_ResNet50_using_keras.ipynb
        # Not using PCA will result in significantly longer t-SNE operation.
        logger.info('\tStarting to Visualise CNN Codes - PCA + t-SNE')
        # Start with reducing dimensions to 500 using PCA
        pca = PCA(n_components=args.PCA_REDUCTION)
        X_training_reduced = pca.fit_transform(X_training)
        logger.info('\tPCA done. Starting t-SNE')
        # # If dataset will be too large uncomment below
        # np.sum(pca.explained_variance_ratio_)

        # Now use t-SNE to visualize in 2 dimensions
        tsne = TSNE(n_components=2, verbose=3, n_iter=args.TSNE_ITERATIONS)
        X_training_reduced_tsne = tsne.fit_transform(X_training_reduced)
        logger.info('\tt-SNE fitting done.')
        plt.figure(figsize=(20, 20))
        # plt.scatter(X_training_reduced_tsne[:, 0], X_training_reduced_tsne[:, 1], c=y_training)

        points = [{'x': [], 'y': []} for _ in range(NUM_CLASSES)]
        for i in range(len(y_training)):
            points[y_training[i]]['x'].append(X_training_reduced_tsne[i, 0])
            points[y_training[i]]['y'].append(X_training_reduced_tsne[i, 1])
        for i in range(NUM_CLASSES):
            color = np.random.rand(3, )
            for single_point in range(len(points[i]['x'])):
                x = points[i]['x'][single_point]
                y = points[i]['y'][single_point]
                # print(x, y, end='\r')
                plt.scatter(x, y, c=color)
        plt.legend()
        plt.savefig(EXPERIMENT_DIR + h5_file[0] + '.jpg')
        # plt.show()
